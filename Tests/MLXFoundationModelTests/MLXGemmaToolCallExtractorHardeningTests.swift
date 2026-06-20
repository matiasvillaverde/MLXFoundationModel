import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX Gemma tool call extractor hardening")
struct MLXGemmaToolCallExtractorHardeningTests {
    @Test("extracts arguments with apostrophes and nested values")
    func extractsArgumentsWithApostrophesAndNestedValues() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|tool_call>\
            call:create_note(title='Bob's note', metadata={tags:['mlx','swift'], count:2})\
            <tool_call|>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let metadata = try #require(arguments["metadata"] as? [String: Any])

        #expect(call.name == "create_note")
        #expect(arguments["title"] as? String == "Bob's note")
        #expect(metadata["tags"] as? [String] == ["mlx", "swift"])
        #expect(metadata["count"] as? Int == 2)
    }

    @Test("extracts namespaced parenthesized call")
    func extractsNamespacedParenthesizedCall() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|tool_call>\
            call:google:mcp:text_generation:create-pdf-file(title='Q2', options={pages:2})\
            <tool_call|>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)
        let options = try #require(arguments["options"] as? [String: Any])

        #expect(call.name == "google:mcp:text_generation:create-pdf-file")
        #expect(arguments["title"] as? String == "Q2")
        #expect(options["pages"] as? Int == 2)
    }

    @Test("extracts degenerate Gemma calls with missing prefix pieces")
    func extractsDegenerateGemmaCallsWithMissingPrefixPieces() throws {
        let calls = MLXToolCallExtractor.extractAll(
            from: """
            <|tool_call>\
            calldone{answer: ok} :reflect(reason=<|"|>complete<|"|>)\
            <tool_call|>
            """
        )
        let first = try #require(calls.first)
        let second = try #require(calls.dropFirst().first)
        let firstArguments = try Self.jsonObject(from: first.argumentsJSON)
        let secondArguments = try Self.jsonObject(from: second.argumentsJSON)

        #expect(first.name == "done")
        #expect(firstArguments["answer"] as? String == "ok")
        #expect(second.name == "reflect")
        #expect(secondArguments["reason"] as? String == "complete")
    }

    @Test("extracts Gemma bare markdown values with commas")
    func extractsGemmaBareMarkdownValuesWithCommas() throws {
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            <|tool_call>\
            calldone{answer:# Title

            | A | B |
            | :--- | :--- |
            | x, y | z |
            ,directive_compliance:1. ok. 2. fine.
            ,memory_ids:[mm-abc123],mental_model_ids:[],observation_ids:[]}\
            <tool_call|>
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "done")
        #expect((arguments["answer"] as? String)?.contains("| x, y | z |") == true)
        #expect(arguments["directive_compliance"] as? String == "1. ok. 2. fine.")
        #expect(arguments["memory_ids"] as? [String] == ["mm-abc123"])
        #expect(arguments["mental_model_ids"] is [Any])
        #expect(arguments["observation_ids"] is [Any])
    }

    @Test("drops oversized arguments cleanly")
    func dropsOversizedArgumentsCleanly() {
        let oversized = String(repeating: "x", count: 270_000)
        let calls = MLXToolCallExtractor.extractAll(
            from: "<|tool_call>call:write{content:'\(oversized)'}<tool_call|>"
        )

        #expect(calls.isEmpty)
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
