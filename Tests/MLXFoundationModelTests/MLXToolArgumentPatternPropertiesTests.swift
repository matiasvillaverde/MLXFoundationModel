import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument patternProperties")
struct MLXToolArgumentPatternPropertiesTests {
    @Test("normalizes root and nested patternProperties schemas")
    func normalizesRootAndNestedPatternPropertiesSchemas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=store>\
            <parameter=x_count>4</parameter>\
            <parameter=enabled_debug>yes</parameter>\
            <parameter=metadata>{"int_limit":"5","flag_trace":"true","text_code":123,"other":"6"}</parameter>\
            </function></tool_call>
            """,
            tools: [Self.patternPropertiesTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let metadata = try #require(arguments["metadata"] as? [String: Any])

        #expect(arguments["x_count"] as? Int == 4)
        #expect(arguments["enabled_debug"] as? Bool == true)
        #expect(metadata["int_limit"] as? Int == 5)
        #expect(metadata["flag_trace"] as? Bool == true)
        #expect(metadata["text_code"] as? String == "123")
        #expect(metadata["other"] as? String == "6")
    }

    private static var patternPropertiesTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "store",
            description: "Store metadata",
            parametersJSONSchema: Self.patternPropertiesSchema
        )
    }

    private static var patternPropertiesSchema: String {
        [
            #"{"type":"object","$defs":{"Flag":{"type":"boolean"}},"#,
            #""patternProperties":{"^x_":{"type":"integer"},"#,
            ##""^enabled_":{"$ref":"#/$defs/Flag"}},"##,
            #""properties":{"metadata":{"type":"object","patternProperties":{"#,
            #""^int_":{"type":"integer"},"#,
            ##""^flag_":{"$ref":"#/$defs/Flag"},"##,
            #""^text_":{"type":"string"}}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
