import Foundation
import Hub
@testable import MLXLocalModels
import Testing

@Suite("RWKV7 tokenizer support")
struct RWKV7TokenizerSupportTests {
    @Test("loads longest-match tokenizer")
    func loadsLongestMatchTokenizer() async throws {
        let directory = try Self.makeTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [5])
        #expect(tokenizer.decode(tokens: [5]) == "hello")
        #expect(tokenizer.convertTokenToId("<|endoftext|>") == 0)
        #expect(tokenizer.decode(tokens: [0, 5], skipSpecialTokens: true) == "hello")
    }

    private static func makeTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "RWKV7Tokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try #"{"<|endoftext|>":0,"h":1,"e":2,"hell":3,"o":4,"hello":5}"#.write(
            to: directory.appending(component: RWKV7Tokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"model":{"type":"RWKV7LongestMatch"}}"#.write(
            to: directory.appending(component: "tokenizer.json"),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"Rwkv7Tokenizer"}"#.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }
}
