#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import CoreGraphics
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsTranscriptBridge {
    struct Result {
        var instructions: [String] = []
        var messages: [MLXBridgeMessage] = []
        var images: [CGImage] = []
        var unsupportedEntries: [Transcript.Entry] = []

        var containsImageAttachments: Bool {
            !images.isEmpty
        }
    }

    static func convert(_ transcript: Transcript) -> Result {
        var result = Result()
        for entry in transcript {
            append(entry, to: &result)
        }
        return result
    }

    private static func append(_ entry: Transcript.Entry, to result: inout Result) {
        switch entry {
        case .instructions(let value):
            appendInstructions(entry, value.segments, to: &result)

        case .prompt(let value):
            appendMessage(entry, value.segments, role: .user, to: &result)

        case .response(let value):
            appendMessage(entry, value.segments, role: .assistant, to: &result)

        case .toolCalls(let calls):
            appendToolCalls(calls, to: &result)

        case .toolOutput(let output):
            appendMessage(entry, output.segments, role: .tool, to: &result, name: output.toolName)

        case .reasoning(let reasoning):
            appendReasoning(entry, reasoning.segments, to: &result)

        @unknown default:
            return
        }
    }

    private static func appendInstructions(
        _ entry: Transcript.Entry,
        _ segments: [Transcript.Segment],
        to result: inout Result
    ) {
        guard let text = text(of: segments, to: &result) else {
            result.unsupportedEntries.append(entry)
            return
        }
        result.instructions.append(text)
    }

    private static func appendMessage(
        _ entry: Transcript.Entry,
        _ segments: [Transcript.Segment],
        role: MLXBridgeRole,
        to result: inout Result,
        name: String? = nil
    ) {
        guard let text = text(of: segments, to: &result) else {
            result.unsupportedEntries.append(entry)
            return
        }
        result.messages.append(MLXBridgeMessage(role: role, content: text, name: name))
    }

    private static func appendReasoning(
        _ entry: Transcript.Entry,
        _ segments: [Transcript.Segment],
        to result: inout Result
    ) {
        guard let text = text(of: segments, to: &result) else {
            result.unsupportedEntries.append(entry)
            return
        }
        guard !text.isEmpty else {
            return
        }
        result.messages.append(MLXBridgeMessage(
            role: .assistant,
            content: "Reasoning:\n\(text)"
        ))
    }

    private static func appendToolCalls(_ calls: Transcript.ToolCalls, to result: inout Result) {
        result.messages.append(MLXBridgeMessage(
            role: .assistant,
            content: FoundationModelsToolSchemaBuilder.toolCallsText(calls)
        ))
    }

    private static func text(of segments: [Transcript.Segment], to result: inout Result) -> String? {
        var hasUnsupportedSegment = false
        let text = segments.compactMap { segment in
            text(of: segment, to: &result, hasUnsupportedSegment: &hasUnsupportedSegment)
        }
        .joined(separator: "\n")
        return hasUnsupportedSegment ? nil : text
    }

    private static func text(
        of segment: Transcript.Segment,
        to result: inout Result,
        hasUnsupportedSegment: inout Bool
    ) -> String? {
        switch segment {
        case .text(let text):
            return text.content

        case .structure(let structure):
            return structure.content.jsonString

        case .attachment(let attachment):
            return text(of: attachment, to: &result, hasUnsupportedSegment: &hasUnsupportedSegment)

        case .custom(let segment):
            let description = String(describing: segment)
            return description.isEmpty ? nil : description

        @unknown default:
            hasUnsupportedSegment = true
            return nil
        }
    }

    private static func text(
        of attachment: Transcript.AttachmentSegment,
        to result: inout Result,
        hasUnsupportedSegment: inout Bool
    ) -> String? {
        switch attachment.content {
        case .image(let image):
            result.images.append(image.cgImage)
            return imagePlaceholder(for: attachment)

        @unknown default:
            hasUnsupportedSegment = true
            return nil
        }
    }

    private static func imagePlaceholder(for attachment: Transcript.AttachmentSegment) -> String {
        if let label = attachment.label, !label.isEmpty {
            return "[Image: \(label)]"
        }
        return "[Image attachment: \(attachment.id)]"
    }
}
#endif
