import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument propertyNames")
struct MLXToolArgumentPropertyNamesTests {
    @Test("selects then branch when propertyNames pattern matches")
    func selectsThenBranchWhenPropertyNamesPatternMatches() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route_keys>\
            <parameter=x_count>"3"</parameter>\
            <parameter=x_flag>yes</parameter>\
            <parameter=x_unused>drop me</parameter>\
            </function></tool_call>
            """,
            tools: [Self.propertyNamesTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["x_count"] as? Int == 3)
        #expect(arguments["x_flag"] as? Bool == true)
        #expect(arguments["x_unused"] == nil)
    }

    @Test("selects else branch when propertyNames pattern fails")
    func selectsElseBranchWhenPropertyNamesPatternFails() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=route_keys>\
            <parameter=count>"3"</parameter>\
            <parameter=fallback>yes</parameter>\
            <parameter=limit>"2"</parameter>\
            </function></tool_call>
            """,
            tools: [Self.propertyNamesTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["count"] == nil)
        #expect(arguments["fallback"] as? Bool == true)
        #expect(arguments["limit"] as? Int == 2)
    }

    private static var propertyNamesTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "route_keys",
            description: "Route by argument keys",
            parametersJSONSchema: Self.propertyNamesSchema
        )
    }

    private static var propertyNamesSchema: String {
        [
            #"{"type":"object","additionalProperties":false,"#,
            #""if":{"propertyNames":{"type":"string","pattern":"^x_","minLength":3}},"#,
            #""then":{"properties":{"x_count":{"type":"integer"},"x_flag":{"type":"boolean"}}},"#,
            #""else":{"properties":{"fallback":{"type":"boolean"},"limit":{"type":"integer"}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
