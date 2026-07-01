import Foundation
import Hub
@testable import MLXLocalModels
import Testing
import Tokenizers

@Suite("Hunyuan tokenizer support")
struct HunyuanTokenizerSupportTests {
    @Test("loads Hunyuan tiktoken tokenizer without tokenizer JSON")
    func loadsHunyuanTiktokenTokenizer() async throws {
        let directory = try Self.makeTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [128])
        #expect(tokenizer.decode(tokens: [130, 128, 132], skipSpecialTokens: true) == "hello")
        #expect(tokenizer.convertTokenToId("<|startoftext|>") == 130)
        #expect(tokenizer.convertTokenToId("<|eos|>") == 132)

        let messages: [Message] = [["role": "user", "content": "hello"]]
        let chatTokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(chatTokens.contains(128))
        #expect(chatTokens.contains(130))
        #expect(chatTokens.contains(134))
        #expect(chatTokens.contains(138))
    }

    private static func makeTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "HunyuanTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Self.tiktokenFixture().write(
            to: directory.appending(component: HunyuanTiktokenTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"HYTokenizer","model_max_length":1048576}"#.write(
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
