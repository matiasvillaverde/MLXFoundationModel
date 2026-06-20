import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX tool argument prefixItems normalization")
struct MLXToolArgumentPrefixItemsTests {
    @Test("normalizes tuple arrays from prefixItems")
    func normalizesTupleArraysFromPrefixItems() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <tool_call><function=plot>\
            <parameter=point>["1","true",3]</parameter>\
            <parameter=tail>[4,"5","6"]</parameter>\
            </function></tool_call>
            """,
            tools: [Self.plotTool]
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let point = try #require(arguments["point"] as? [Any])
        let tail = try #require(arguments["tail"] as? [Any])

        #expect(point.count == 3)
        #expect(point.first as? Int == 1)
        #expect(point[1] as? Bool == true)
        #expect(point.last as? String == "3")

        #expect(tail.count == 3)
        #expect(tail.first as? String == "4")
        #expect(tail[1] as? Int == 5)
        #expect(tail.last as? Int == 6)
    }

    private static var plotTool: MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: "plot",
            description: "Plot a point",
            parametersJSONSchema: Self.plotSchema
        )
    }

    private static var plotSchema: String {
        [
            #"{"type":"object","properties":{"#,
            #""point":{"type":"array","prefixItems":["#,
            #"{"type":"integer"},{"type":"boolean"},{"type":"string"}]},"#,
            #""tail":{"type":"array","prefixItems":[{"type":"string"}],"#,
            #""items":{"type":"integer"}}}}"#
        ].joined()
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
