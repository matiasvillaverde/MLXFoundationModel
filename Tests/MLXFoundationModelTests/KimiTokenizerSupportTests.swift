import Foundation
import Hub
@testable import MLXLocalModels
import Testing
import Tokenizers

@Suite("Kimi tokenizer support")
struct KimiTokenizerSupportTests {
    @Test("loads Kimi tiktoken tokenizer without tokenizer JSON")
    func loadsKimiTiktokenTokenizer() async throws {
        let directory = try Self.makeKimiTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [128])
        #expect(tokenizer.decode(tokens: [128]) == "hello")
        #expect(tokenizer.convertTokenToId("<|im_user|>") == 163_587)
        #expect(tokenizer.decode(tokens: [163_587, 128, 163_586], skipSpecialTokens: true) == "hello")

        let messages: [Message] = [["role": "user", "content": "hello"]]
        let chatTokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(chatTokens.contains(163_587))
        #expect(chatTokens.contains(163_601))
        #expect(chatTokens.contains(163_586))
        #expect(chatTokens.contains(128))
    }

    private static func makeKimiTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "KimiTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Self.tiktokenFixture().write(
            to: directory.appending(component: KimiTiktokenTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"TikTokenTokenizer","model_max_length":8192}"#.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func tiktokenFixture() -> String {
        var vocab = (0 ... 127).map { byte in
            (Data([UInt8(byte)]), byte)
        }
        vocab.append((Data("hello".utf8), 128))
        return vocab
            .map { token, rank in "\(token.base64EncodedString()) \(rank)" }
            .joined(separator: "\n")
    }
}
