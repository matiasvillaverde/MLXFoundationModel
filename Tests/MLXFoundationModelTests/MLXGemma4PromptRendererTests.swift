import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Gemma 4 prompt renderer")
struct MLXGemma4PromptRendererTests {
    @Test("renders native Gemma 4 turns and tool declarations")
    func rendersNativeTurnsAndToolDeclarations() {
        let rendered = MLXPromptRenderer.render(Self.userRequest, style: .gemma)

        #expect(rendered.prompt.hasPrefix("<bos><|turn>system\n"))
        #expect(rendered.prompt.contains("<|tool>declaration:weather{"))
        #expect(rendered.prompt.contains(#"description:<|"|>Read weather<|"|>"#))
        #expect(rendered.prompt.contains(#"type:<|"|>OBJECT<|"|>"#))
        #expect(rendered.prompt.contains("<|turn>user\nWhat is the weather?<turn|>"))
        #expect(rendered.prompt.hasSuffix("<|turn>model\n"))
        #expect(!rendered.prompt.contains("<start_of_turn>"))
    }

    @Test("folds tool results onto the Gemma 4 model turn")
    func foldsToolResultsOntoModelTurn() {
        let rendered = MLXPromptRenderer.render(Self.toolResultRequest, style: .gemma)

        #expect(rendered.prompt.contains(#"<|tool_call>call:weather{city:<|"|>Berlin<|"|>}<tool_call|>"#))
        #expect(rendered.prompt.contains(
            #"<|tool_response>response:weather{temperature:18,weather:<|"|>sunny<|"|>}<tool_response|>"#
        ))
        #expect(!rendered.prompt.contains("Tool weather:"))
        #expect(!rendered.prompt.contains("<|turn>user\nTool weather"))
        #expect(rendered.prompt.hasSuffix("<tool_response|>"))
    }

    @Test("continues the same model turn after a tool response")
    func continuesSameModelTurnAfterToolResponse() {
        let rendered = MLXPromptRenderer.render(Self.finalAnswerRequest, style: .gemma)
        let expected = """
        <tool_response|>The weather is sunny.<turn|>
        <|turn>model
        """

        #expect(rendered.prompt.contains(expected))
        #expect(Self.occurrences(of: "<|turn>model\n", in: rendered.prompt) == 2)
    }

    @Test("renders assistant reasoning history as native thought channel")
    func rendersAssistantReasoningHistoryAsNativeThoughtChannel() {
        let rendered = MLXPromptRenderer.render(Self.reasoningHistoryRequest, style: .gemma)
        let expected = """
        <|turn>model
        <|channel>thought
        Checked the tool result.<channel|>Final.
        """

        #expect(rendered.prompt.contains(expected))
        #expect(rendered.prompt.contains("Final.<turn|>"))
        #expect(!rendered.prompt.contains("Reasoning:"))
    }

    @Test("renders reasoning history without requiring a tool call")
    func rendersReasoningHistoryWithoutRequiringToolCall() {
        let rendered = MLXPromptRenderer.render(Self.reasoningOnlyHistoryRequest, style: .gemma)
        let expected = """
        <|turn>model
        <|channel>thought
        Considered the answer.<channel|><turn|>
        """

        #expect(rendered.prompt.contains(expected))
        #expect(!rendered.prompt.contains("<|tool_call>"))
        #expect(!rendered.prompt.contains("Reasoning:"))
    }

    @Test("enriches tool schemas without mutating caller definitions")
    func enrichesToolSchemasWithoutMutatingCallerDefinitions() {
        let tool = Self.delegateTool
        let rendered = MLXPromptRenderer.render(Self.delegateRequest(tool: tool), style: .gemma)
        let descriptionMarker = #"""
        description:<|"|>REQUIRED. The 'description' value (type: string)<|"|>
        """#

        #expect(rendered.prompt.contains("param_description:{"))
        #expect(rendered.prompt.contains(descriptionMarker))
        #expect(!tool.parametersJSONSchema.contains("param_description"))
    }

    private static var userRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [MLXBridgeMessage(role: .user, content: "What is the weather?")],
            tools: [weatherTool]
        )
    }

    private static var toolResultRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
                ),
                MLXBridgeMessage(
                    role: .tool,
                    content: #"{"temperature":18,"weather":"sunny"}"#,
                    name: "weather"
                )
            ],
            tools: [weatherTool]
        )
    }

    private static var finalAnswerRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: toolResultRequest.messages + [
                MLXBridgeMessage(role: .assistant, content: "The weather is sunny.")
            ],
            tools: [weatherTool]
        )
    }

    private static var reasoningHistoryRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: "Final.",
                    reasoningContent: "Checked the tool result."
                )
            ]
        )
    }

    private static var reasoningOnlyHistoryRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .assistant,
                    content: "",
                    reasoningContent: "Considered the answer."
                )
            ]
        )
    }

    private static func delegateRequest(tool: MLXBridgeToolDefinition) -> MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(role: .user, content: "Delegate.")
            ],
            tools: [tool]
        )
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: """
            {"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}
            """
        )
    }

    private static var delegateTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "delegate",
            description: "",
            parametersJSONSchema: """
            {"type":"object","properties":{"description":{"type":"string"},\
            "prompt":{"type":"string","description":"Prompt text"}},\
            "required":["description","prompt"]}
            """
        )
    }

    private static func occurrences(of marker: String, in text: String) -> Int {
        text.components(separatedBy: marker).count - 1
    }
}
