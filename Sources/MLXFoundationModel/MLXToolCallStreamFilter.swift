import Foundation

struct MLXToolCallStreamFilter {
    private enum DrainStep {
        case continueDraining
        case stopDraining
    }

    private var buffer = ""
    private var activeEnd: String?
    private var consumesBareGLMToolCall = false
    private var consumesMistralToolCall = false
    private var suppressedText = ""
    private var completedSuppressedTexts: [String] = []
    private let toolNames: Set<String>

    init(toolNames: Set<String> = []) {
        self.toolNames = toolNames
    }

    mutating func feed(_ text: String) -> String {
        buffer += text
        return drain(final: false)
    }

    mutating func finish() -> String {
        drain(final: true)
    }

    mutating func takeCompletedSuppressedTexts() -> [String] {
        defer {
            completedSuppressedTexts.removeAll(keepingCapacity: true)
        }
        return completedSuppressedTexts
    }

    private mutating func drain(final: Bool) -> String {
        var output = ""
        while true {
            if let step = drainActiveSuppression(final: final) {
                if step == .stopDraining {
                    return output
                }
                continue
            }
            let envelope = MLXToolCallEnvelopeDetector.firstEnvelope(in: buffer)
            if let bareGLMRange = firstBareGLMRange(before: envelope) {
                output += String(buffer[..<bareGLMRange.lowerBound])
                let startText = String(buffer[bareGLMRange])
                buffer = String(buffer[bareGLMRange.upperBound...])
                activateBareGLM(startText: startText)
                continue
            }
            guard let envelope else {
                output += drainVisibleBuffer(final: final)
                return output
            }
            output += String(buffer[..<envelope.range.lowerBound])
            let startText = String(buffer[envelope.range])
            buffer = String(buffer[envelope.range.upperBound...])
            activate(envelope, startText: startText)
        }
    }

    private mutating func drainActiveSuppression(final: Bool) -> DrainStep? {
        if consumesBareGLMToolCall {
            return consumeBareGLMToolCall(final: final) ? .continueDraining : .stopDraining
        }
        if consumesMistralToolCall {
            return consumeMistralToolCall(final: final) ? .continueDraining : .stopDraining
        }
        if let activeEnd {
            return consumeSuppressedContent(until: activeEnd, final: final)
                ? .continueDraining
                : .stopDraining
        }
        return nil
    }

    private mutating func activate(
        _ envelope: MLXToolCallEnvelopeDetector.Envelope,
        startText: String
    ) {
        switch envelope.kind {
        case .consumeOnly:
            completedSuppressedTexts.append(startText)
            return

        case .mistral:
            consumesMistralToolCall = true
            suppressedText = startText

        case .paired(let end):
            activeEnd = end
            suppressedText = startText
        }
    }

    private mutating func activateBareGLM(startText: String) {
        consumesBareGLMToolCall = true
        suppressedText = startText
    }

    private func firstBareGLMRange(
        before envelope: MLXToolCallEnvelopeDetector.Envelope?
    ) -> Range<String.Index>? {
        guard let range = MLXBareGLMToolCallScanner.firstStart(in: buffer, toolNames: toolNames) else {
            return nil
        }
        guard let envelope else {
            return range
        }
        return range.lowerBound < envelope.range.lowerBound ? range : nil
    }

    private mutating func consumeSuppressedContent(
        until marker: String,
        final: Bool
    ) -> Bool {
        guard let range = buffer.range(of: marker) else {
            if final {
                suppressedText = ""
                buffer = ""
                activeEnd = nil
            } else {
                let retained = buffer.suffixString(retainingAtMost: marker.count - 1)
                suppressedText += String(buffer.dropLast(retained.count))
                buffer = retained
            }
            return false
        }
        suppressedText += String(buffer[..<range.upperBound])
        completedSuppressedTexts.append(suppressedText)
        buffer = String(buffer[range.upperBound...])
        activeEnd = nil
        suppressedText = ""
        return true
    }

    private mutating func consumeBareGLMToolCall(final: Bool) -> Bool {
        let combined = suppressedText + buffer
        guard let range = MLXBareGLMToolCallScanner.completedRange(
            in: combined,
            toolNames: toolNames,
            final: final
        ) else {
            if final {
                buffer = ""
                consumesBareGLMToolCall = false
                suppressedText = ""
            }
            return false
        }
        completedSuppressedTexts.append(String(combined[range]))
        buffer = String(combined[range.upperBound...])
        consumesBareGLMToolCall = false
        suppressedText = ""
        return true
    }

    private mutating func consumeMistralToolCall(final: Bool) -> Bool {
        guard let range = MLXMistralToolCallEnvelopeScanner.range(in: buffer) else {
            if final {
                buffer = ""
                consumesMistralToolCall = false
                suppressedText = ""
            }
            return false
        }
        suppressedText += String(buffer[..<range.upperBound])
        completedSuppressedTexts.append(suppressedText)
        buffer = String(buffer[range.upperBound...])
        consumesMistralToolCall = false
        suppressedText = ""
        return true
    }

    private mutating func drainVisibleBuffer(final: Bool) -> String {
        guard !final else {
            defer {
                buffer = ""
            }
            return finalVisibleBuffer()
        }
        let protocolRetainCount = MLXToolCallEnvelopeDetector.partialStartSuffixLength(in: buffer)
        let bareGLMRetainCount = MLXBareGLMToolCallScanner.partialStartSuffixLength(
            in: buffer,
            toolNames: toolNames
        ) ?? 0
        let retainCount = max(protocolRetainCount, bareGLMRetainCount)
        guard buffer.count > retainCount else {
            return ""
        }
        let split = retainCount == 0
            ? buffer.endIndex
            : buffer.index(buffer.endIndex, offsetBy: -retainCount)
        let visible = String(buffer[..<split])
        buffer = String(buffer[split...])
        return visible
    }

    private func finalVisibleBuffer() -> String {
        let protocolRetainCount = MLXToolCallEnvelopeDetector.partialStartSuffixLength(in: buffer)
        let bareGLMRetainCount = MLXBareGLMToolCallScanner.partialStartSuffixLength(
            in: buffer,
            toolNames: toolNames
        ) ?? 0
        let retainCount = max(protocolRetainCount, bareGLMRetainCount)
        guard retainCount > 0 else {
            return buffer
        }

        let tailStart = buffer.index(buffer.endIndex, offsetBy: -retainCount)
        let prefix = String(buffer[..<tailStart])
        let tail = String(buffer[tailStart...])
        return MLXToolCallEnvelopeDetector.shouldDropUnresolvedTail(tail)
            || MLXBareGLMToolCallScanner.shouldDropUnresolvedTail(tail, toolNames: toolNames)
            ? prefix
            : buffer
    }
}
