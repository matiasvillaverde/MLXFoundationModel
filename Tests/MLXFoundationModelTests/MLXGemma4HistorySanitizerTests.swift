import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Gemma 4 history sanitizer")
struct MLXGemma4HistorySanitizerTests {
    @Test("strips leading rendered thinking from Gemma history")
    func stripsLeadingRenderedThinkingFromGemmaHistory() {
        let rendered = MLXPromptRenderer.render(
            Self.request(assistantContent: "<think>reasoning</think>\nAnswer"),
            style: .gemma
        )

        #expect(rendered.prompt.contains("Answer"))
        #expect(!rendered.prompt.contains("<think>"))
        #expect(!rendered.prompt.contains("reasoning"))
    }

    @Test("strips leading raw Gemma channel block from history")
    func stripsLeadingRawGemmaChannelBlockFromHistory() {
        let rendered = MLXPromptRenderer.render(
            Self.request(assistantContent: "<|channel>thought\nreasoning<channel|>Answer"),
            style: .gemma
        )

        #expect(rendered.prompt.contains("Answer"))
        #expect(!rendered.prompt.contains("<|channel>"))
        #expect(!rendered.prompt.contains("<channel|>"))
        #expect(!rendered.prompt.contains("reasoning"))
    }

    @Test("strips stray Gemma tool markers without dropping valid tool-call history")
    func stripsStrayGemmaToolMarkersWithoutDroppingValidToolCallHistory() {
        let stray = MLXPromptRenderer.render(
            Self.request(assistantContent: "<tool_call|>"),
            style: .gemma
        )
        let valid = MLXPromptRenderer.render(
            Self.request(assistantContent: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#),
            style: .gemma
        )

        #expect(!stray.prompt.contains("<tool_call|>"))
        #expect(valid.prompt.contains("<|tool_call>call:weather"))
        #expect(valid.prompt.contains("<tool_call|>"))
    }

    @Test("keeps inline thinking tag mentions outside the leading block")
    func keepsInlineThinkingTagMentionsOutsideLeadingBlock() {
        let rendered = MLXPromptRenderer.render(
            Self.request(assistantContent: "Use <think> only as a label."),
            style: .gemma
        )

        #expect(rendered.prompt.contains("Use <think> only as a label."))
    }

    private static func request(assistantContent: String) -> MLXBridgeRequest {
        MLXBridgeRequest(messages: [
            MLXBridgeMessage(role: .assistant, content: assistantContent)
        ])
    }
}
