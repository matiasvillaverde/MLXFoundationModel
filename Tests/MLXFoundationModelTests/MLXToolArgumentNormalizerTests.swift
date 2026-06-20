import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument normalizer")
struct MLXToolArgumentNormalizerTests {
    @Test("normalizes extracted XML arguments with tool schemas")
    func normalizesExtractedXMLArgumentsWithToolSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather>\
            <parameter=code>123</parameter>\
            <parameter=count>2</parameter>\
            <parameter=enabled>true</parameter>\
            <parameter=payload>{"limit":"4"}</parameter>\
            <parameter=tags>["1","2"]</parameter>\
            </function></tool_call>
            """,
            tools: [Self.weatherTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])
        let tags = try #require(arguments["tags"] as? [Int])

        #expect(arguments["code"] as? String == "123")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(payload["limit"] as? Int == 4)
        #expect(tags == [1, 2])
    }

    @Test("normalizes referenced and branch object schemas")
    func normalizesReferencedAndBranchObjectSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather>\
            <parameter=payload>{"limit":"4","unit":123}</parameter>\
            <parameter=items>[{"count":"1"},{"count":"2"}]</parameter>\
            <parameter=mode>7</parameter>\
            </function></tool_call>
            """,
            tools: [Self.referencedSchemaTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])
        let items = try #require(arguments["items"] as? [[String: Any]])

        #expect(payload["limit"] as? Int == 4)
        #expect(payload["unit"] as? String == "123")
        #expect(items.first?["count"] as? Int == 1)
        #expect(items.last?["count"] as? Int == 2)
        #expect(arguments["mode"] as? String == "7")
    }

    @Test("leaves unknown tool arguments unchanged")
    func leavesUnknownToolArgumentsUnchanged() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<tool_call><function=other><parameter=count>2</parameter></function></tool_call>"#,
            tools: [Self.weatherTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["count"] as? Int == 2)
    }

    @Test("remaps boundary-aligned namespaced tool names")
    func remapsBoundaryAlignedNamespacedToolNames() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:google:mcp:text_generation:create-pdf-file{"count":"2"}<tool_call|>"#,
            tools: [Self.createPDFTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "create-pdf-file")
        #expect(arguments["count"] as? Int == 2)
    }

    @Test("keeps ambiguous namespaced tool names unchanged")
    func keepsAmbiguousNamespacedToolNamesUnchanged() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:google:mcp:text_generation:create-pdf-file{"count":"2"}<tool_call|>"#,
            tools: [
                Self.createPDFTool,
                Self.namespacedCreatePDFTool
            ]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "google:mcp:text_generation:create-pdf-file")
        #expect(arguments["count"] as? String == "2")
    }

    @Test("restores Gemma template aliases before schema normalization")
    func restoresGemmaTemplateAliasesBeforeSchemaNormalization() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"<|tool_call>call:delegate{param_description:'audit',prompt:'review'}<tool_call|>"#,
            tools: [Self.delegateTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["description"] as? String == "audit")
        #expect(arguments["prompt"] as? String == "review")
        #expect(arguments["param_description"] == nil)
    }

    @Test("normalizes object additionalProperties schemas")
    func normalizesObjectAdditionalPropertiesSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=store>\
            <parameter=metadata>{"limit":"4","retries":"2"}</parameter>\
            <parameter=debug>yes</parameter>\
            </function></tool_call>
            """,
            tools: [Self.dynamicPropertiesTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let metadata = try #require(arguments["metadata"] as? [String: Int])

        #expect(metadata == ["limit": 4, "retries": 2])
        #expect(arguments["debug"] as? Bool == true)
    }

    @Test("infers types from const and nullable schema hints")
    func infersTypesFromConstAndNullableSchemaHints() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route>\
            <parameter=kind>weather</parameter>\
            <parameter=count>"2"</parameter>\
            <parameter=enabled>"true"</parameter>\
            <parameter=note>nil</parameter>\
            </function></tool_call>
            """,
            tools: [Self.constTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["kind"] as? String == "weather")
        #expect(arguments["count"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == true)
        #expect(arguments["note"] is NSNull)
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: Self.weatherSchema
        )
    }

    private static var createPDFTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "create-pdf-file",
            description: "Create a PDF",
            parametersJSONSchema: Self.countSchema
        )
    }

    private static var namespacedCreatePDFTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "text_generation:create-pdf-file",
            description: "Create a PDF",
            parametersJSONSchema: Self.countSchema
        )
    }

    private static var referencedSchemaTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: Self.referencedSchema
        )
    }

    private static var delegateTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "delegate",
            description: "Delegate work",
            parametersJSONSchema: Self.delegateSchema
        )
    }

    private static var dynamicPropertiesTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "store",
            description: "Store metadata",
            parametersJSONSchema: Self.dynamicPropertiesSchema
        )
    }

    private static var constTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "route",
            description: "Route a request",
            parametersJSONSchema: Self.constSchema
        )
    }

    private static var weatherSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""code":{"type":"string"},"#,
            #""count":{"type":"integer"},"#,
            #""enabled":{"type":"boolean"},"#,
            #""payload":{"type":"object","properties":{"limit":{"type":"integer"}}},"#,
            #""tags":{"type":"array","items":{"type":"integer"}}}}"#
        ].joined()
    }

    private static var countSchema: String {
        #"{"type":"object","properties":{"count":{"type":"integer"}}}"#
    }

    private static var referencedSchema: String {
        [
            #"{"type":"object","$defs":{"#,
            #""Payload":{"type":"object","properties":{"limit":{"type":"integer"},"#,
            #""unit":{"type":"string"}}},"#,
            #""Item":{"type":"object","properties":{"count":{"type":"integer"}}}},"#,
            ##""properties":{"payload":{"anyOf":[{"$ref":"#/$defs/Payload"},{"type":"null"}]},"##,
            ##""items":{"type":"array","items":{"$ref":"#/$defs/Item"}},"##,
            #""mode":{"anyOf":[{"type":"string"},{"type":"integer"}]}}}"#
        ].joined()
    }

    private static var delegateSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""description":{"type":"string"},"#,
            #""prompt":{"type":"string"}},"#,
            #""required":["description","prompt"]}"#
        ].joined()
    }

    private static var dynamicPropertiesSchema: String {
        [
            #"{"type":"object","additionalProperties":{"type":"boolean"},"#,
            #""properties":{"metadata":{"type":"object","additionalProperties":{"type":"integer"}}}}"#
        ].joined()
    }

    private static var constSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""kind":{"const":"weather"},"#,
            #""count":{"const":2},"#,
            #""enabled":{"const":true},"#,
            #""note":{"type":"string","nullable":true}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
