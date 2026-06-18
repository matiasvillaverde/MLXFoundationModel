#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsTranscriptBridge {
    struct Result {
        var instructions: [String] = []
        var messages: [MLXBridgeMessage] = []
        var unsupportedEntries: [Transcript.Entry] = []
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

        case .reasoning:
            return

        @unknown default:
            return
        }
    }

    private static func appendInstructions(
        _ entry: Transcript.Entry,
        _ segments: [Transcript.Segment],
        to result: inout Result
    ) {
        guard let text = text(of: segments) else {
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
        guard let text = text(of: segments) else {
            result.unsupportedEntries.append(entry)
            return
        }
        result.messages.append(MLXBridgeMessage(role: role, content: text, name: name))
    }

    private static func appendToolCalls(_ calls: Transcript.ToolCalls, to result: inout Result) {
        result.messages.append(MLXBridgeMessage(
            role: .assistant,
            content: FoundationModelsToolSchemaBuilder.toolCallsText(calls)
        ))
    }

    private static func text(of segments: [Transcript.Segment]) -> String? {
        var hasUnsupportedSegment = false
        let text = segments.compactMap { segment in
            text(of: segment, hasUnsupportedSegment: &hasUnsupportedSegment)
        }
        .joined(separator: "\n")
        return hasUnsupportedSegment ? nil : text
    }

    private static func text(
        of segment: Transcript.Segment,
        hasUnsupportedSegment: inout Bool
    ) -> String? {
        switch segment {
        case .text(let text):
            return text.content

        case .structure(let structure):
            return structure.content.jsonString

        case .attachment, .custom:
            hasUnsupportedSegment = true
            return nil

        @unknown default:
            hasUnsupportedSegment = true
            return nil
        }
    }
}
#endif
