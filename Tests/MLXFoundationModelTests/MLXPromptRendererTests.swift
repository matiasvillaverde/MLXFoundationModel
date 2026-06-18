import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX prompt renderer")
struct MLXPromptRendererTests {
    @Test("renders ChatML prompt with sorted tool definitions")
    func rendersChatMLPromptWithTools() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "What is the weather?")
            ],
            instructions: "Be concise.",
            tools: [
                MLXBridgeToolDefinition(
                    name: "weather",
                    description: "Read local weather",
                    parametersJSONSchema: #"{"type":"object"}"#
                )
            ]
        )

        let rendered = MLXPromptRenderer.render(request, style: .chatML)

        #expect(rendered.rendererID == "mlx.chatML.v1")
        #expect(rendered.prompt.contains("<|im_start|>system"))
        #expect(rendered.prompt.contains("Available tools:"))
        #expect(rendered.prompt.contains("- weather: Read local weather"))
        #expect(rendered.prompt.contains("<|im_start|>user\nWhat is the weather?<|im_end|>"))
        #expect(rendered.prompt.hasSuffix("<|im_start|>assistant\n"))
        #expect(!rendered.cacheFingerprint.isEmpty)
    }

    @Test("renders plain prompt with assistant continuation")
    func rendersPlainPrompt() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Explain MLX.")
            ],
            instructions: "Answer as a sentence."
        )

        let rendered = MLXPromptRenderer.render(request, style: .plain)

        #expect(rendered.prompt.contains("System:\nAnswer as a sentence."))
        #expect(rendered.prompt.contains("User:\nExplain MLX."))
        #expect(rendered.prompt.hasSuffix("Assistant:"))
    }
}
