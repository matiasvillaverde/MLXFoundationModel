import Foundation
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelToolCallingTests {
    @Test("Qwen3 tool stream normalizes arguments with tool schemas")
    func qwen3ToolStreamNormalizesArgumentsWithToolSchemas() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.schemaNormalizationRequest,
                limits: ResourceLimits(maxTokens: 180, maxTime: .seconds(120), reusePromptCache: false)
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let call = try #require(Self.normalizedToolCalls(in: observed.result).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])
        let tags = try #require(arguments["tags"] as? [Int])

        Self.verifyGeneratedTokens(tokenEvents)
        #expect(call.name == "weather")
        #expect(arguments["code"] as? String == "123")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(payload["limit"] as? Int == 4)
        #expect(tags == [1, 2])
        #expect(Self.responseTexts(in: observed.result).allSatisfy { !$0.contains("<tool_call") })
    }

    @Test("Qwen3 constrained native tool stream emits typed arguments")
    func qwen3ConstrainedNativeToolStreamEmitsTypedArguments() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.schemaConstrainedNativeToolRequest,
                limits: ResourceLimits(maxTokens: 80, maxTime: .seconds(120), reusePromptCache: false),
                style: .qwenXML,
                sampling: Self.schemaConstrainedNativeToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let call = try #require(Self.normalizedToolCalls(in: observed.result).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        Self.verifyGeneratedTokens(tokenEvents)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(call.name == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(Self.responseTexts(in: observed.result).allSatisfy { !$0.contains("<tool_call") })
    }

    @Test("Qwen3 constrained tool stream selects oneOf payload branch")
    func qwen3ConstrainedToolStreamSelectsOneOfPayloadBranch() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }
        let observed = try await MLXGenerationDiagnostics.withRecording {
            try await MLXRealModelHarness.translateRenderedRequest(
                model: model,
                request: Self.oneOfBranchToolRequest,
                limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false),
                style: .qwenXML,
                sampling: Self.oneOfBranchToolSampling
            )
        }
        let tokenEvents = MLXRealModelHarness.generatedTokenSnapshots(from: observed.events)
        let grammarEvents = MLXRealModelHarness.grammarSnapshots(from: observed.events)
        let call = try #require(Self.normalizedToolCalls(in: observed.result).first)
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])

        Self.verifyGeneratedTokens(tokenEvents)
        Self.verifySuccessfulGrammarDiagnostics(grammarEvents, kind: .ebnf)
        #expect(call.name == "route")
        #expect(payload["kind"] as? String == "search")
        #expect(payload["query"] as? String == "123")
        #expect(payload["limit"] as? Int == 4)
        #expect(payload["path"] == nil)
        #expect(payload["secure"] == nil)
        #expect(Self.responseTexts(in: observed.result).allSatisfy { !$0.contains("<tool_call") })
    }

    private static var schemaNormalizationRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nCall the weather tool with the exact values from the instructions."
                )
            ],
            instructions: """
            Emit only this Qwen XML tool call and no prose:
            <tool_call><function=weather>\
            <parameter=code>123</parameter>\
            <parameter=count>2</parameter>\
            <parameter=enabled>true</parameter>\
            <parameter=payload>{"limit":"4"}</parameter>\
            <parameter=tags>["1","2"]</parameter>\
            </function></tool_call>
            """,
            tools: [schemaWeatherTool]
        )
    }

    private static var schemaConstrainedNativeToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nCall the weather tool with count 2 and enabled true."
                )
            ],
            instructions: """
            Emit exactly one Qwen XML tool call and no prose.
            The count argument must be 2. The enabled argument must be true.
            """,
            tools: [schemaConstrainedNativeTool]
        )
    }

    private static var oneOfBranchToolRequest: MLXBridgeRequest {
        MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nCall the route tool with the exact payload from the instructions."
                )
            ],
            instructions: """
            Emit exactly one Qwen XML tool call and no prose.
            The payload must use kind search, query 123, limit 4, and no open-route fields.
            """,
            tools: [oneOfBranchTool]
        )
    }

    private static var schemaConstrainedNativeTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather controls",
            parametersJSONSchema: schemaConstrainedNativeToolParameters
        )
    }

    private static var oneOfBranchTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "route",
            description: "Route a request",
            parametersJSONSchema: oneOfBranchToolParameters
        )
    }

    private static var schemaConstrainedNativeToolParameters: String {
        """
        {"type":"object","required":["count","enabled"],\
        "properties":{"count":{"type":"integer"},"enabled":{"type":"boolean"}}}
        """
    }

    private static var oneOfBranchToolParameters: String {
        [
            #"{"type":"object","required":["payload"],"properties":{"payload":{"oneOf":["#,
            #"{"type":"object","additionalProperties":false,"required":["kind"],"properties":{"#,
            #""kind":{"const":"search"},"query":{"type":"string"},"limit":{"type":"integer"}}},"#,
            #"{"type":"object","additionalProperties":false,"required":["kind"],"properties":{"#,
            #""kind":{"const":"open"},"path":{"type":"string"},"secure":{"type":"boolean"}}}"#,
            #"]}}}"#
        ].joined()
    }

    private static var schemaConstrainedNativeToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: schemaConstrainedNativeToolGrammar)
            )
        )
    }

    private static var oneOfBranchToolSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: oneOfBranchToolGrammar)
            )
        )
    }

    private static let schemaConstrainedNativeToolGrammar = """
    root ::= "<tool_call><function=weather><parameter=count>" fm_json_integer \
    "</parameter><parameter=enabled>" fm_json_boolean "</parameter></function></tool_call>"
    fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*
    fm_json_boolean ::= "true" | "false"
    """

    private static let oneOfBranchToolGrammar = """
    root ::= "<tool_call><function=route><parameter=payload>" \
    "{\\"kind\\":\\"search\\",\\"query\\":123,\\"limit\\":\\"4\\"," \
    "\\"path\\":456,\\"secure\\":\\"true\\"}</parameter></function></tool_call>"
    """

    private static var schemaWeatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read local weather for a city code",
            parametersJSONSchema: schemaWeatherToolParameters
        )
    }

    private static var schemaWeatherToolParameters: String {
        [
            #"{"type":"object","required":["code","count","enabled","payload","tags"],"#,
            #""properties":{"#,
            #""code":{"type":"string"},"#,
            #""count":{"type":"integer"},"#,
            #""enabled":{"type":"boolean"},"#,
            #""payload":{"type":"object","properties":{"limit":{"type":"integer"}}},"#,
            #""tags":{"type":"array","items":{"type":"integer"}}}}"#
        ].joined()
    }

    private static func normalizedToolCalls(
        in events: [MLXTranslatedStreamEvent]
    ) -> [MLXExtractedToolCall] {
        events.compactMap { event in
            guard case .toolCall(let call, _) = event else {
                return nil
            }
            return call
        }
    }

    private static func responseTexts(
        in events: [MLXTranslatedStreamEvent]
    ) -> [String] {
        events.compactMap { event in
            guard case .responseText(let text, _) = event else {
                return nil
            }
            return text
        }
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private static func verifySuccessfulGrammarDiagnostics(
        _ events: [MLXGrammarConstraintSnapshot],
        kind: GrammarConstraintKind
    ) {
        let summary = Comment(rawValue: grammarEventSummary(events))
        #expect(events.contains { $0.stage == .matcherPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .maskApplied && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .mlxMaskPrepared && $0.kind == kind }, summary)
        #expect(events.contains { $0.stage == .tokenAccepted && $0.kind == kind }, summary)
        #expect(!events.contains { $0.stage == .tokenRejected }, summary)
        #expect(!events.contains { $0.stage == .processorFailedClosed }, summary)
    }

    private static func grammarEventSummary(_ events: [MLXGrammarConstraintSnapshot]) -> String {
        events
            .map { event in
                [
                    "stage=\(event.stage.rawValue)",
                    "kind=\(event.kind.map(String.init(describing:)) ?? "nil")",
                    "mode=\(event.mode.map(String.init(describing:)) ?? "nil")",
                    "tokenCount=\(event.tokenCount.map(String.init) ?? "nil")",
                    "tokenID=\(event.tokenID.map(String.init) ?? "nil")",
                    "message=\(event.message ?? "nil")"
                ].joined(separator: " ")
            }
            .joined(separator: "\n")
    }

    private static func verifyGeneratedTokens(
        _ tokens: [MLXGeneratedTokenSnapshot]
    ) {
        let summary = Comment(rawValue: tokens.map(\.tokenText).joined())
        #expect(!tokens.isEmpty, summary)
        #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }
}
