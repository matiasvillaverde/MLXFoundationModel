import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX prompt renderer")
struct MLXPromptRendererTests {
    @Test("renders ChatML prompt with sorted tool definitions")
    func rendersChatMLPromptWithTools() throws {
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
        let toolRange = try #require(rendered.prompt.range(of: "Available tools:"))
        let instructionRange = try #require(rendered.prompt.range(of: "Be concise."))
        #expect(toolRange.lowerBound < instructionRange.lowerBound)
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

    @Test("renders structured response constraints")
    func rendersStructuredResponseConstraints() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Return a city forecast.")
            ],
            responseConstraint: MLXBridgeResponseConstraint(
                jsonSchema: #"{"type":"object","required":["city"]}"#
            )
        )

        let rendered = MLXPromptRenderer.render(request, style: .chatML)

        #expect(rendered.prompt.contains("Response constraints:"))
        #expect(rendered.prompt.contains("Return only JSON that conforms to this schema."))
        #expect(rendered.prompt.contains(#""required":["city"]"#))
    }

    @Test("cache fingerprint tracks renderer compatibility instead of prompt text")
    func cacheFingerprintTracksRendererCompatibility() {
        let first = MLXPromptRenderer.render(
            MLXBridgeRequest(messages: [
                MLXBridgeMessage(role: .user, content: "First prompt")
            ]),
            style: .chatML
        )
        let second = MLXPromptRenderer.render(
            MLXBridgeRequest(messages: [
                MLXBridgeMessage(role: .user, content: "Second prompt")
            ]),
            style: .chatML
        )
        let plain = MLXPromptRenderer.render(
            MLXBridgeRequest(messages: [
                MLXBridgeMessage(role: .user, content: "First prompt")
            ]),
            style: .plain
        )

        #expect(first.cacheFingerprint == second.cacheFingerprint)
        #expect(first.cacheFingerprint != plain.cacheFingerprint)
    }
}
