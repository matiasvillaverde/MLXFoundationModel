import Foundation
import Hub
@testable import MLXLocalModels
import Testing
import Tokenizers

@Suite("Plamo tokenizer support")
struct PlamoTokenizerSupportTests {
    @Test("loads Plamo JSONL tokenizer without tokenizer JSON")
    func loadsPlamoTokenizer() async throws {
        let directory = try Self.makePlamoTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.hasChatTemplate == false)
        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [260])
        #expect(tokenizer.encode(text: "hello").first == 1)
        #expect(tokenizer.decode(tokens: [260], skipSpecialTokens: true) == "hello")
        #expect(tokenizer.decode(tokens: [1, 260, 2], skipSpecialTokens: true) == "hello")
        #expect(tokenizer.convertTokenToId("<|plamo:eos|>") == 2)

        let exclamationID = try #require(tokenizer.convertTokenToId("<0x21>"))
        #expect(tokenizer.encode(text: "hello!", addSpecialTokens: false) == [260, exclamationID])
        #expect(tokenizer.decode(tokens: [260, exclamationID], skipSpecialTokens: true) == "hello!")

        let vocabulary = try PlamoTokenizer.vocabularyEntries(
            from: directory.appending(component: PlamoTokenizer.vocabFilename)
        )
        #expect(vocabulary[260] == "hello")
        #expect(vocabulary[exclamationID] == "!")

        #expect(throws: Tokenizers.TokenizerError.self) {
            try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "hello"]])
        }
    }

    private static func makePlamoTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "PlamoTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Self.plamoJSONLFixture().write(
            to: directory.appending(component: PlamoTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"PlamoTokenizer","add_bos_token":true,"add_eos_token":false}"#.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func plamoJSONLFixture() throws -> String {
        var rows: [[Any]] = [
            ["<|plamo:unk|>", 0.0, "UNKNOWN"],
            ["<|plamo:bos|>", 0.0, "CONTROL"],
            ["<|plamo:eos|>", 0.0, "CONTROL"],
            ["<|plamo:pad|>", 0.0, "CONTROL"]
        ]
        rows.append(contentsOf: (0 ... 255).map { byte in
            [String(format: "<0x%02X>", byte), 0.0, "BYTE"] as [Any]
        })
        rows.append(["hello", 2.0, "NORMAL"])
        rows.append(["he", 1.0, "NORMAL"])
        rows.append(["h", 0.1, "NORMAL"])
        rows.append(["e", 0.1, "NORMAL"])
        rows.append(["l", 0.1, "NORMAL"])
        rows.append(["o", 0.1, "NORMAL"])

        return try rows.map { row in
            let data = try JSONSerialization.data(withJSONObject: row)
            guard let line = String(data: data, encoding: .utf8) else {
                throw FixtureError.invalidJSONLine
            }
            return line
        }
        .joined(separator: "\n")
    }

    private enum FixtureError: Error {
        case invalidJSONLine
    }
}
