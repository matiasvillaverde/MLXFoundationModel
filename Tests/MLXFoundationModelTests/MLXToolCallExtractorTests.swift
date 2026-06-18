import MLXFoundationModel
import Testing

@Suite("MLX tool call extractor")
struct MLXToolCallExtractorTests {
    @Test("extracts flat JSON tool call")
    func extractsFlatJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: #"{"tool_name":"weather","arguments":{"city":"Berlin"}}"#
        ))

        #expect(call.name == "weather")
        #expect(call.argumentsJSON == #"{"city":"Berlin"}"#)
    }

    @Test("extracts fenced JSON tool call")
    func extractsFencedJSONToolCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            ```json
            {"name":"search","arguments":{"query":"MLX Swift"}}
            ```
            """
        ))

        #expect(call.name == "search")
        #expect(call.argumentsJSON == #"{"query":"MLX Swift"}"#)
    }

    @Test("ignores normal assistant prose")
    func ignoresNormalAssistantProse() {
        let call = MLXToolCallExtractor.extract(from: "I can answer without tools.")

        #expect(call == nil)
    }
}
