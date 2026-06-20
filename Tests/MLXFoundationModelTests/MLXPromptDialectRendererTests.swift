import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX prompt dialect renderer")
struct MLXPromptDialectRendererTests {
    @Test("renders native tool instructions for model dialects")
    func rendersNativeToolInstructionsForModelDialects() {
        let markers: [MLXPromptStyle: String] = [
            .qwenXML: "<tool_call><function=weather>",
            .glmXML: "<arg_key>city</arg_key>",
            .minimaxXML: "<minimax:tool_call><invoke name=\"weather\">",
            .minimaxM3: "]<]minimax[>[<tool_call>",
            .gemma: "<|tool>declaration:weather",
            .functionGemma: "<start_function_call>call:weather",
            .harmony: "to=functions.weather<|channel|>commentary<|message|>",
            .kimiK2: "tool_declare",
            .longCat: "<longcat_tool_call>",
            .cohereAction: "<|START_ACTION|>",
            .mistralToolCall: "[TOOL_CALLS]weather[ARGS]"
        ]

        for (style, marker) in markers {
            let rendered = MLXPromptRenderer.render(Self.request, style: style)

            #expect(rendered.prompt.contains(marker))
            #expect(rendered.prompt.contains(toolHeader(for: style)))
            #expect(rendered.rendererID == "mlx.\(style.codingValue).v1")
        }
    }

    @Test("uses structured native tool declarations for XML and marker dialects")
    func usesStructuredNativeToolDeclarationsForXMLAndMarkerDialects() {
        let styles: [MLXPromptStyle] = [
            .cohereAction,
            .functionGemma,
            .glmXML,
            .harmony,
            .minimaxXML,
            .mistralToolCall,
            .qwenXML
        ]

        for style in styles {
            let rendered = MLXPromptRenderer.render(Self.request, style: style)

            #expect(!rendered.prompt.contains("Available tools:"))
            #expect(rendered.prompt.contains(Self.weatherTool.parametersJSONSchema) == false)
            #expect(rendered.prompt.contains(#""type":"function""#))
        }

        let gemma = MLXPromptRenderer.render(Self.request, style: .gemma)
        #expect(gemma.prompt.contains("<|tool>declaration:weather"))
        #expect(!gemma.prompt.contains("Available tools:"))
        #expect(!gemma.prompt.contains(#""type":"function""#))
    }

    @Test("replays assistant tool-call history in the selected dialect")
    func replaysAssistantToolCallHistoryInSelectedDialect() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .minimaxXML)

        #expect(rendered.prompt.contains(#"<invoke name="weather">"#))
        #expect(rendered.prompt.contains(#"<parameter name="city">Berlin</parameter>"#))
    }

    @Test("replays Kimi K2 tool-call history with section markers")
    func replaysKimiK2ToolCallHistoryWithSectionMarkers() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .kimiK2)

        #expect(rendered.prompt.contains("<|tool_calls_section_begin|>"))
        #expect(rendered.prompt.contains(#"<|tool_call_begin|>functions.weather:0"#))
    }

    @Test("replays MiniMax M3 tool-call history with native namespace tokens")
    func replaysMiniMaxM3ToolCallHistoryWithNativeNamespaceTokens() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .minimaxM3)

        #expect(rendered.prompt.contains(#"]<]minimax[>[<invoke name="weather">"#))
        #expect(rendered.prompt.contains(#"]<]minimax[>[<city>Berlin]<]minimax[>[</city>"#))
    }

    @Test("replays LongCat tool-call history with JSON payloads")
    func replaysLongCatToolCallHistoryWithJSONPayloads() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin","count":2}}"#
                )
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .longCat)
        let expectedCall = """
        <longcat_tool_call>{"arguments":{"city":"Berlin","count":2},"name":"weather"}</longcat_tool_call>
        """

        #expect(rendered.prompt.contains(expectedCall))
        #expect(!rendered.prompt.contains("<longcat_arg_key>"))
    }

    @Test("continues LongCat assistant turn after tool result")
    func continuesLongCatAssistantTurnAfterToolResult() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                ),
                MLXBridgeMessage(role: .tool, content: #"{"temperature":18}"#, name: "weather")
            ],
            reasoningEnabled: true,
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .longCat)

        #expect(rendered.prompt.contains("TOOL:Tool weather:\n"))
        #expect(rendered.prompt.contains(#"{"temperature":18}</longcat_s> "#))
        #expect(rendered.prompt.hasSuffix("ASSISTANT:<longcat_think>\n"))
    }

    @Test("uses native role templates for non-ChatML dialects")
    func usesNativeRoleTemplatesForNonChatMLDialects() {
        let expectedMarkers: [MLXPromptStyle: [String]] = [
            .glmXML: ["[gMASK]<sop><|system|>", "<|user|>", "<|assistant|>"],
            .gemma: ["<|turn>user\n", "<|turn>model\n"],
            .functionGemma: ["<start_of_turn>user", "<start_of_turn>model"],
            .harmony: [
                "<|start|>system<|message|>",
                "<|start|>user<|message|>",
                "<|start|>assistant"
            ],
            .mistralToolCall: ["[SYSTEM_PROMPT]", "[INST]What is the weather?[/INST]"],
            .cohereAction: ["<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>", "<|CHATBOT_TOKEN|>"],
            .kimiK2: ["<|im_system|>tool_declare<|im_middle|>", "<|im_user|>user<|im_middle|>"],
            .longCat: ["## Messages", "SYSTEM:Be concise.", "[Round 0] USER:What is the weather? ASSISTANT:"],
            .minimaxM3: ["]~!b[]~b]system\n", "]~b]developer\n", "]~b]user\n", "]~b]ai\n"],
            .minimaxXML: ["]~!b[]~b]system\n", "]~b]user\n", "]~b]ai\n<think>\n"]
        ]

        for (style, markers) in expectedMarkers {
            let rendered = MLXPromptRenderer.render(Self.requestWithInstructions, style: style)

            for marker in markers {
                #expect(rendered.prompt.contains(marker))
            }
            #expect(!rendered.prompt.contains("<|im_start|>"))
        }
    }

    @Test("keeps Qwen on ChatML role markers")
    func keepsQwenOnChatMLRoleMarkers() {
        let rendered = MLXPromptRenderer.render(Self.requestWithInstructions, style: .qwenXML)

        #expect(rendered.prompt.contains("<|im_start|>system"))
        #expect(rendered.prompt.contains("<|im_start|>user\nWhat is the weather?<|im_end|>"))
        #expect(rendered.prompt.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test("renders Harmony tool-call and tool-result history")
    func rendersHarmonyToolCallAndToolResultHistory() {
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                ),
                MLXBridgeMessage(role: .tool, content: #"{"temperature":18}"#, name: "weather")
            ],
            tools: [Self.weatherTool]
        )

        let rendered = MLXPromptRenderer.render(request, style: .harmony)

        #expect(rendered.prompt.contains(#"<|start|>assistant to=functions.weather"#))
        #expect(rendered.prompt.contains(#"<|message|>{"city":"Berlin"}<|call|>"#))
        #expect(rendered.prompt.contains(#"<|start|>functions.weather to=assistant"#))
        #expect(rendered.prompt.contains(#"<|message|>{"temperature":18}<|end|>"#))
        #expect(rendered.prompt.hasSuffix("<|start|>assistant"))
    }

    @Test("adds template-safe descriptions to tool schemas")
    func addsTemplateSafeDescriptionsToToolSchemas() {
        let rendered = MLXPromptRenderer.render(Self.missingDescriptionRequest, style: .qwenXML)

        #expect(rendered.prompt.contains(#""description":"""#))
        #expect(rendered.prompt.contains(#""query":{"description":"","type":"string"}"#))
        #expect(rendered.prompt.contains(#""limit":{"description":"","type":"integer"}"#))
    }

    private static var request: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "What is the weather?")
            ],
            tools: [weatherTool]
        )
    }

    private static var requestWithInstructions: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "What is the weather?")
            ],
            instructions: "Be concise.",
            tools: [weatherTool]
        )
    }

    private static var missingDescriptionRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Search.")
            ],
            tools: [searchTool]
        )
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather",
            parametersJSONSchema: """
            {"type":"object","properties":{"city":{"type":"string"},"count":{"type":"integer"}}}
            """
        )
    }

    private static var searchTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "search",
            description: "",
            parametersJSONSchema: """
            {"type":"object","properties":{"query":{"type":"string"},\
            "filters":{"type":"object","properties":{"limit":{"type":"integer"}}}}}
            """
        )
    }

    private func toolHeader(for style: MLXPromptStyle) -> String {
        switch style {
        case .cohereAction, .functionGemma, .glmXML, .minimaxXML, .qwenXML:
            return "# Tools"

        case .gemma:
            return "<|tool>declaration"

        case .kimiK2:
            return "tool_declare"

        case .mistralToolCall:
            return "[AVAILABLE_TOOLS]"

        case .harmony:
            return "# Tools"

        case .longCat, .minimaxM3:
            return style == .longCat ? "## Tools" : "# Tools"

        default:
            return "Available tools:"
        }
    }
}
