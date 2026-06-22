import Foundation
import MLXFoundationModel
import Testing

@Suite("MLX JSON text extractor")
struct MLXJSONTextExtractorTests {
    @Test("extracts large embedded JSON object without prefix reparsing")
    func extractsLargeEmbeddedJSONObjectWithoutPrefixReparsing() throws {
        let text = String(repeating: "x", count: 16_384)
        let call = try #require(MLXToolCallExtractor.extract(
            from: """
            prefix {not json}
            {"tool_name":"echo","arguments":{"text":"\(text)"}}
            """
        ))
        let arguments = try Self.jsonObject(from: call.argumentsJSON)

        #expect(call.name == "echo")
        #expect(arguments["text"] as? String == text)
    }

    private static func jsonObject(from text: String) throws -> [String: Any] {
        let data = try #require(text.data(using: .utf8))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
