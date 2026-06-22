import Foundation

struct MLXHarmonyStreamFilter {
    private enum ContentMode {
        case visible
        case hidden
        case thinking
    }

    private struct ContentDrain {
        let output: String
        let shouldContinue: Bool
    }

    private static let activationStarts = specialMarkers + [" to=functions."]
    private static let specialMarkers = [
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "<|call|>",
        "<|constrain|>"
    ]
    private static let contentTerminators = [
        "<|end|>",
        "<|return|>",
        "<|call|>"
    ]
    private static let thinkOpen = "<think>\n"
    private static let thinkClose = "</think>\n"

    private var buffer = ""
    private var isHarmonyMode = false
    private var currentContentMode: ContentMode?
    private var shouldDropLeadingReplacement = false

    mutating func feed(_ text: String) -> String {
        buffer += String.consumingLeadingUnicodeReplacementIfNeeded(
            from: text,
            shouldDrop: &shouldDropLeadingReplacement
        )
        return drain(final: false)
    }

    mutating func finish() -> String {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> String {
        var output = ""
        while true {
            if !isHarmonyMode {
                guard let activationRange = firstActivationRange(in: buffer) else {
                    output += drainPlainText(final: final)
                    return output
                }
                output += String(buffer[..<activationRange.lowerBound])
                    .droppingTrailingUnicodeReplacementCharacter()
                buffer = String(buffer[activationRange.lowerBound...])
                isHarmonyMode = true
            }

            if let currentContentMode {
                let result = drainContent(mode: currentContentMode, final: final)
                output += result.output
                if result.shouldContinue {
                    continue
                }
                return output
            }

            guard let headerOutput = consumeProtocolHeader(final: final) else {
                return output
            }
            output += headerOutput
        }
    }

    private mutating func consumeProtocolHeader(final: Bool) -> String? {
        guard let channelRange = buffer.range(of: "<|channel|>") else {
            if final {
                buffer = ""
            } else {
                buffer = buffer.suffixString(retainingAtMost: longestActivationPrefixLength - 1)
            }
            return nil
        }

        guard let messageRange = buffer.range(
            of: "<|message|>",
            range: channelRange.upperBound..<buffer.endIndex
        ) else {
            if final {
                buffer = ""
            }
            return nil
        }

        let channelDescriptor = String(buffer[channelRange.upperBound..<messageRange.lowerBound])
        let channel = Self.channelName(in: channelDescriptor)
        let mode = Self.contentMode(channel: channel)
        currentContentMode = mode
        buffer = String(buffer[messageRange.upperBound...])
        String.dropLeadingUnicodeReplacement(
            from: &buffer,
            orNextChunk: &shouldDropLeadingReplacement
        )
        return mode == .thinking ? Self.thinkOpen : ""
    }

    private mutating func drainContent(
        mode: ContentMode,
        final: Bool
    ) -> ContentDrain {
        if let terminatorRange = firstTerminatorRange(in: buffer) {
            return drainTerminatedContent(mode: mode, terminatorRange: terminatorRange)
        }

        guard !final else {
            return drainFinalContent(mode: mode)
        }

        let retainCount = partialSuffixLength(in: buffer, markers: Self.contentTerminators)
        if mode == .hidden {
            buffer = retainCount > 0 ? String(buffer.suffix(retainCount)) : ""
            return ContentDrain(output: "", shouldContinue: false)
        }

        guard buffer.count > retainCount else {
            return ContentDrain(output: "", shouldContinue: false)
        }
        let split = retainCount == 0
            ? buffer.endIndex
            : buffer.index(buffer.endIndex, offsetBy: -retainCount)
        var visible = String(buffer[..<split])
        if retainCount > 0 {
            visible = visible.droppingTrailingUnicodeReplacementCharacter()
        }
        buffer = String(buffer[split...])
        return ContentDrain(output: visible, shouldContinue: false)
    }

    private mutating func drainTerminatedContent(
        mode: ContentMode,
        terminatorRange: Range<String.Index>
    ) -> ContentDrain {
        let content = String(buffer[..<terminatorRange.lowerBound])
            .droppingTrailingUnicodeReplacementCharacter()
        buffer = String(buffer[terminatorRange.upperBound...])
        String.dropLeadingUnicodeReplacement(
            from: &buffer,
            orNextChunk: &shouldDropLeadingReplacement
        )
        currentContentMode = nil
        return ContentDrain(
            output: output(for: content, mode: mode, final: true),
            shouldContinue: true
        )
    }

    private mutating func drainFinalContent(mode: ContentMode) -> ContentDrain {
        defer {
            buffer = ""
            currentContentMode = nil
        }
        return ContentDrain(
            output: output(for: buffer, mode: mode, final: true),
            shouldContinue: false
        )
    }

    private func output(
        for content: String,
        mode: ContentMode,
        final: Bool
    ) -> String {
        switch mode {
        case .visible:
            return content

        case .hidden:
            return ""

        case .thinking:
            return final ? content + Self.thinkClose : content
        }
    }

    private mutating func drainPlainText(final: Bool) -> String {
        guard !final else {
            defer { buffer = "" }
            return buffer
        }
        let retainCount = partialSuffixLength(in: buffer, markers: Self.activationStarts)
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

    private func firstActivationRange(in text: String) -> Range<String.Index>? {
        firstRange(of: Self.activationStarts, in: text)
    }

    private func firstTerminatorRange(in text: String) -> Range<String.Index>? {
        firstRange(of: Self.contentTerminators, in: text)
    }

    private func firstRange(
        of markers: [String],
        in text: String
    ) -> Range<String.Index>? {
        markers
            .compactMap { text.range(of: $0) }
            .min { $0.lowerBound < $1.lowerBound }
    }

    private func partialSuffixLength(
        in text: String,
        markers: [String]
    ) -> Int {
        let longestMarkerLength = markers.map(\.count).max() ?? 0
        let maximum = max(0, min(text.count, longestMarkerLength - 1))
        guard maximum > 0 else {
            return 0
        }
        for length in stride(from: maximum, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            guard suffix.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false else {
                continue
            }
            if markers.contains(where: { $0.hasPrefix(suffix) }) {
                return length
            }
        }
        return 0
    }

    private var longestActivationPrefixLength: Int {
        Self.activationStarts.map(\.count).max() ?? 0
    }

    private static func contentMode(channel: String?) -> ContentMode {
        switch channel {
        case "final":
            return .visible

        case "analysis":
            return .thinking

        case "commentary":
            return .hidden

        default:
            return .visible
        }
    }

    static func channelName(in text: String) -> String? {
        text
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(whereSeparator: \.isWhitespace)
            .first
            .map(String.init)
    }

    static func recipient(in text: String) -> String? {
        guard let range = text.range(of: #"to=functions\.([A-Za-z_][\w.:-]*)"#, options: .regularExpression)
        else {
            return nil
        }
        let match = String(text[range])
        return String(match.dropFirst("to=".count))
    }
}
