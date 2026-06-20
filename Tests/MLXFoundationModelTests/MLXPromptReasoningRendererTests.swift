import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX prompt reasoning renderer")
struct MLXPromptReasoningRendererTests {
    @Test("opens native reasoning channels when requested")
    func opensNativeReasoningChannelsWhenRequested() {
        let markers: [MLXPromptStyle: String] = [
            .chatML: "<|im_start|>assistant\n<think>\n",
            .cohereAction: "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|>",
            .deepSeekDSML: "<｜Assistant｜><think>\n",
            .gemma: "<|turn>model\n<|channel>thought\n",
            .glmXML: "<|assistant|><think>\n",
            .harmony: "<|start|>assistant<|channel|>analysis<|message|>",
            .longCat: "[Round 0] USER:Answer after thinking. ASSISTANT:<longcat_think>\n",
            .minimaxM3: "]~b]ai\n<mm:think>",
            .qwenXML: "<|im_start|>assistant\n<think>\n"
        ]

        for (style, marker) in markers {
            let rendered = MLXPromptRenderer.render(Self.reasoningRequest, style: style)

            #expect(rendered.prompt.contains(marker))
        }
    }

    @Test("structured reasoning options drive native thinking markers")
    func structuredReasoningOptionsDriveNativeThinkingMarkers() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Answer after thinking.")
            ],
            reasoningOptions: .enabled(effort: .deep)
        )

        let rendered = MLXPromptRenderer.render(request, style: .minimaxM3)

        #expect(request.reasoningEnabled)
        #expect(request.effectiveReasoningOptions.effort == .deep)
        #expect(rendered.prompt.hasSuffix("]~b]ai\n<mm:think>"))
    }

    @Test("decodes legacy reasoning enabled bridge requests")
    func decodesLegacyReasoningEnabledBridgeRequests() throws {
        let json = """
        {
            "messages": [],
            "reasoningEnabled": true,
            "tools": []
        }
        """
        let data = Data(json.utf8)

        let request = try JSONDecoder().decode(MLXBridgeRequest.self, from: data)

        #expect(request.reasoningOptions == nil)
        #expect(request.effectiveReasoningOptions.isEnabled)
    }

    @Test("normalizes custom reasoning effort")
    func normalizesCustomReasoningEffort() {
        let options = MLXBridgeReasoningOptions.enabled(customEffort: "  careful  ")

        #expect(options.isEnabled)
        #expect(options.customEffort == "careful")
    }

    private static var reasoningRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Answer after thinking.")
            ],
            reasoningEnabled: true
        )
    }
}
