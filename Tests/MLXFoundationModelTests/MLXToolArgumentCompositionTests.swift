import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument composition normalization")
struct MLXToolArgumentCompositionTests {
    @Test("merges branch properties with sibling object properties")
    func mergesBranchPropertiesWithSiblingObjectProperties() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather>\
            <parameter=city>123</parameter>\
            <parameter=count>"2"</parameter>\
            <parameter=payload>{"unit":456,"limit":"4"}</parameter>\
            </function></tool_call>
            """,
            tools: [Self.mixedCompositionTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])

        #expect(arguments["city"] as? String == "123")
        #expect(arguments["count"] as? Int == 2)
        #expect(payload["unit"] as? String == "456")
        #expect(payload["limit"] as? Int == 4)
    }

    @Test("selects oneOf object branch before normalizing nested arguments")
    func selectsOneOfObjectBranchBeforeNormalizingNestedArguments() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route>\
            <parameter=payload>{"kind":"search","query":123,"limit":"4",\
            "path":456,"secure":"true"}</parameter>\
            </function></tool_call>
            """,
            tools: [Self.oneOfObjectTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let payload = try #require(arguments["payload"] as? [String: Any])

        #expect(payload["kind"] as? String == "search")
        #expect(payload["query"] as? String == "123")
        #expect(payload["limit"] as? Int == 4)
        #expect(payload["path"] == nil)
        #expect(payload["secure"] == nil)
    }

    private static var mixedCompositionTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: Self.mixedCompositionSchema
        )
    }

    private static var oneOfObjectTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "route",
            description: "Route a request",
            parametersJSONSchema: Self.oneOfObjectSchema
        )
    }

    private static var mixedCompositionSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""city":{"type":"string"},"#,
            #""payload":{"type":"object","properties":{"unit":{"type":"string"}},"#,
            #""allOf":[{"properties":{"limit":{"type":"integer"}}}]}},"#,
            #""allOf":[{"properties":{"count":{"type":"integer"}}}]}"#
        ].joined()
    }

    private static var oneOfObjectSchema: String {
        [
            #"{"type":"object","properties":{"payload":{"oneOf":["#,
            #"{"type":"object","additionalProperties":false,"required":["kind"],"properties":{"#,
            #""kind":{"const":"search"},"query":{"type":"string"},"limit":{"type":"integer"}}},"#,
            #"{"type":"object","additionalProperties":false,"required":["kind"],"properties":{"#,
            #""kind":{"const":"open"},"path":{"type":"string"},"secure":{"type":"boolean"}}}"#,
            #"]}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
