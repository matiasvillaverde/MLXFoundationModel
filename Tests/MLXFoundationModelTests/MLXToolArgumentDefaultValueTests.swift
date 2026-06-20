import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument default values")
struct MLXToolArgumentDefaultValueTests {
    @Test("materializes root and nested schema defaults")
    func materializesRootAndNestedSchemaDefaults() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather>\
            <parameter=city>Berlin</parameter>\
            <parameter=options>{"limit":"2"}</parameter>\
            </function></tool_call>
            """,
            tools: [Self.weatherTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let options = try #require(arguments["options"] as? [String: Any])

        #expect(arguments["city"] as? String == "Berlin")
        #expect(arguments["unit"] as? String == "metric")
        #expect(arguments["includeHourly"] as? Bool == false)
        #expect(options["limit"] as? Int == 2)
        #expect(options["format"] as? String == "brief")
    }

    @Test("materializes active branch defaults only")
    func materializesActiveBranchDefaultsOnly() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=weather>\
            <parameter=city>Paris</parameter>\
            <parameter=mode>daily</parameter>\
            </function></tool_call>
            """,
            tools: [Self.weatherTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(arguments["mode"] as? String == "daily")
        #expect(arguments["days"] as? Int == 3)
        #expect(arguments["hours"] == nil)
    }

    private static var weatherTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "weather",
            description: "Read weather",
            parametersJSONSchema: Self.weatherSchema
        )
    }

    private static var weatherSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""city":{"type":"string"},"#,
            #""unit":{"type":"string","default":"metric"},"#,
            #""includeHourly":{"type":"boolean","default":"false"},"#,
            #""mode":{"type":"string"},"#,
            #""options":{"type":"object","properties":{"#,
            #""limit":{"type":"integer"},"#,
            #""format":{"type":"string","default":"brief"}}}},"#,
            #""if":{"properties":{"mode":{"const":"daily"}},"required":["mode"]},"#,
            #""then":{"properties":{"days":{"type":"integer","default":"3"}}},"#,
            #""else":{"properties":{"hours":{"type":"integer","default":"24"}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
