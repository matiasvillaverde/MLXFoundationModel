import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument literal schema normalization")
struct MLXToolArgumentLiteralSchemaTests {
    @Test("canonicalizes enum and const values without forcing mismatches")
    func canonicalizesEnumAndConstValues() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=select>\
            <parameter=mode>"2"</parameter>\
            <parameter=enabled>"false"</parameter>\
            <parameter=fruit>banana</parameter>\
            <parameter=count>"7"</parameter>\
            <parameter=nullish>nil</parameter>\
            <parameter=wrong>traffic</parameter>\
            </function></tool_call>
            """,
            tools: [Self.selectTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["mode"] as? Int == 2)
        #expect(arguments["enabled"] as? Bool == false)
        #expect(arguments["fruit"] as? String == "banana")
        #expect(arguments["count"] as? Int == 7)
        #expect(arguments["nullish"] is NSNull)
        #expect(arguments["wrong"] as? String == "traffic")
    }

    private static var selectTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "select",
            description: "Select a finite value",
            parametersJSONSchema: Self.selectSchema
        )
    }

    private static var selectSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""mode":{"enum":["auto",2,false]},"#,
            #""enabled":{"enum":[true,false]},"#,
            #""fruit":{"anyOf":[{"enum":["apple","pear","banana"]}]},"#,
            #""count":{"const":7},"#,
            #""nullish":{"enum":[null,"none"]},"#,
            #""wrong":{"const":"weather"}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
