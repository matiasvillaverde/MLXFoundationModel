import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument additionalProperties")
struct MLXToolArgumentAdditionalPropertiesTests {
    @Test("drops unknown properties when schema rejects additionalProperties")
    func dropsUnknownPropertiesWhenSchemaRejectsAdditionalProperties() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=store>\
            <parameter=debug>yes</parameter>\
            <parameter=extra>drop me</parameter>\
            <parameter=metadata>{"limit":"4","ignored":"drop me"}</parameter>\
            </function></tool_call>
            """,
            tools: [Self.strictTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let metadata = try #require(arguments["metadata"] as? [String: Any])

        #expect(arguments["debug"] as? Bool == true)
        #expect(arguments["extra"] == nil)
        #expect(metadata["limit"] as? Int == 4)
        #expect(metadata["ignored"] == nil)
    }

    private static var strictTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "store",
            description: "Store metadata",
            parametersJSONSchema: Self.strictSchema
        )
    }

    private static var strictSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"properties":{"#,
            #""debug":{"type":"boolean"},"#,
            #""metadata":{"type":"object","additionalProperties":false,"properties":{"#,
            #""limit":{"type":"integer"}}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
