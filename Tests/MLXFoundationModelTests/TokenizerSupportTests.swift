import Foundation
import Hub
@testable import MLXLocalModels
import Testing
import Tokenizers

@Suite("Tokenizer support")
struct TokenizerSupportTests {
    @Test("rewrites configured tokenizer classes while preserving other config keys")
    func rewritesConfiguredTokenizerClasses() {
        let registry = TokenizerReplacementRegistry(replacements: ["Qwen3Tokenizer": "PreTrainedTokenizer"])
        let rewriter = TokenizerConfigurationRewriter(registry: registry)
        let config = Config([
            "tokenizer_class": Config("Qwen3Tokenizer"),
            "model_max_length": Config(4_096)
        ])

        let rewritten = rewriter.rewrite(config)

        #expect(config.tokenizerClass?.string() == "Qwen3Tokenizer")
        #expect(rewritten.tokenizerClass?.string() == "PreTrainedTokenizer")
        #expect(rewritten.dictionary()?["model_max_length"]?.integer() == 4_096)
    }

    @Test("leaves unsupported tokenizer classes unchanged")
    func leavesUnsupportedTokenizerClassesUnchanged() {
        let registry = TokenizerReplacementRegistry(replacements: ["Qwen3Tokenizer": "PreTrainedTokenizer"])
        let rewriter = TokenizerConfigurationRewriter(registry: registry)
        let config = Config(["tokenizer_class": Config("CustomTokenizer")])

        #expect(rewriter.rewrite(config) == config)
    }

    @Test("default registry rewrites GPT-NeoX tokenizer class")
    func defaultRegistryRewritesGPTNeoXTokenizerClass() {
        let rewriter = TokenizerConfigurationRewriter(registry: TokenizerReplacementRegistry())
        let config = Config(["tokenizer_class": Config("GPTNeoXTokenizer")])

        #expect(rewriter.rewrite(config).tokenizerClass?.string() == "PreTrainedTokenizer")
    }

    @Test("default registry rewrites RWKV7 tokenizer class")
    func defaultRegistryRewritesRWKV7TokenizerClass() {
        let rewriter = TokenizerConfigurationRewriter(registry: TokenizerReplacementRegistry())
        let config = Config(["tokenizer_class": Config("Rwkv7Tokenizer")])

        #expect(rewriter.rewrite(config).tokenizerClass?.string() == "PreTrainedTokenizer")
    }

    @Test("loads Phi-3-small tiktoken tokenizer without tokenizer JSON")
    func loadsPhi3SmallTiktokenTokenizer() async throws {
        let directory = try Self.makePhi3SmallTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.encode(text: "he", addSpecialTokens: false) == [0])
        #expect(tokenizer.decode(tokens: [0]) == "he")
        #expect(tokenizer.convertTokenToId("<|user|>") == 100_262)

        let messages: [Message] = [["role": "user", "content": "he"]]
        let chatTokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(chatTokens.contains(100_257))
        #expect(chatTokens.contains(100_262))
        #expect(chatTokens.contains(100_266))
        #expect(chatTokens.contains(100_263))
        #expect(chatTokens.contains(0))
    }

    @Test("loads Qwen tiktoken tokenizer without tokenizer JSON")
    func loadsQwenTiktokenTokenizer() async throws {
        let directory = try Self.makeQwenTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [128])
        #expect(tokenizer.decode(tokens: [128]) == "hello")
        #expect(tokenizer.convertTokenToId("<|im_start|>") == 151_644)
        #expect(tokenizer.decode(tokens: [151_644, 128, 151_645], skipSpecialTokens: true) == "hello")

        let messages: [Message] = [["role": "user", "content": "hello"]]
        let chatTokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(chatTokens.contains(151_644))
        #expect(chatTokens.contains(151_645))
        #expect(chatTokens.contains(128))
    }

    @Test("loads SentencePiece model tokenizer without tokenizer JSON")
    func loadsSentencePieceModelTokenizer() async throws {
        let directory = try Self.makeSentencePieceTokenizerFixture()
        let tokenizer = try await loadTokenizer(
            configuration: ModelConfiguration(directory: directory),
            hub: HubApi()
        )

        #expect(tokenizer.convertTokenToId("he") == 4)
        #expect(tokenizer.encode(text: "hello", addSpecialTokens: false) == [6])
        #expect(tokenizer.decode(tokens: [6], skipSpecialTokens: true) == "hello")
        #expect(tokenizer.decode(tokens: [10, 6, 11], skipSpecialTokens: true) == "hello")

        let messages: [Message] = [["role": "user", "content": "hello"]]
        let chatTokens = try tokenizer.applyChatTemplate(messages: messages)
        #expect(chatTokens.contains(1))
        #expect(chatTokens.contains(10))
        #expect(chatTokens.contains(11))
        #expect(chatTokens.contains(4))
        #expect(chatTokens.contains(5))
    }

    @Test("routes fast Unigram tokenizers to Unigram implementation")
    func routesFastUnigramTokenizersToUnigramImplementation() {
        let rewriter = TokenizerConfigurationRewriter(registry: TokenizerReplacementRegistry())
        let config = Config(["tokenizer_class": Config("PreTrainedTokenizerFast")])
        let tokenizerData = Config([
            "model": Config([
                "type": Config("Unigram")
            ])
        ])

        #expect(
            rewriter.rewrite(config, tokenizerData: tokenizerData)
                .tokenizerClass?.string() == "XLMRobertaTokenizer"
        )
    }

    @Test("supports registry replacement updates and removals")
    func supportsRegistryUpdatesAndRemovals() {
        let registry = TokenizerReplacementRegistry(replacements: ["A": "B"])

        #expect(registry.replacement(for: "A") == "B")

        registry["A"] = "C"
        registry["D"] = "E"
        #expect(registry.replacement(for: "A") == "C")
        #expect(registry.replacement(for: "D") == "E")

        registry["A"] = nil
        #expect(registry.replacement(for: "A") == nil)
    }

    @Test("streams only newly decoded text and resets after newlines")
    func streamsOnlyNewlyDecodedText() {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: PreparedGenerationTokenizer())

        detokenizer.append(token: 10)
        #expect(detokenizer.next() == "hello")

        detokenizer.append(token: 13)
        #expect(detokenizer.next() == "\n")

        detokenizer.append(token: 11)
        #expect(detokenizer.next() == "world")
    }

    @Test("waits when the decoded suffix ends with a replacement character")
    func waitsForCompleteUnicodeScalar() {
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: PreparedGenerationTokenizer())

        detokenizer.append(token: 12)

        #expect(detokenizer.next() == nil)
    }

    private static func makePhi3SmallTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "Phi3SmallTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let vocab: [(String, Int)] = [
            ("he", 0),
            ("h", 1),
            ("e", 2),
            ("\n", 3),
            (" ", 4),
            ("l", 5),
            ("o", 6)
        ]
        let encodedVocab = vocab
            .map { token, rank in
                "\(Data(token.utf8).base64EncodedString()) \(rank)"
            }
            .joined(separator: "\n")
        try encodedVocab.write(
            to: directory.appending(component: Phi3SmallTiktokenTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"Phi3SmallTokenizer","model_max_length":8192}"#.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func makeQwenTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "QwenTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Self.qwenTiktokenFixture().write(
            to: directory.appending(component: QwenTiktokenTokenizer.vocabFilename),
            atomically: true,
            encoding: .utf8
        )
        try #"{"tokenizer_class":"QWenTokenizer","model_max_length":8192}"#.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )
        return directory
    }

    private static func qwenTiktokenFixture() -> String {
        var vocab = (0 ... 127).map { byte in
            (Data([UInt8(byte)]), byte)
        }
        vocab.append((Data("hello".utf8), 128))
        return vocab
            .map { token, rank in "\(token.base64EncodedString()) \(rank)" }
            .joined(separator: "\n")
    }

    private static func makeSentencePieceTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "SentencePieceTokenizer-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        try Self.makeSentencePieceModel(pieces: sentencePiecePieces).write(
            to: directory.appending(component: SentencePieceModelTokenizer.modelFilename)
        )
        try sentencePieceTokenizerConfig.write(
            to: directory.appending(component: "tokenizer_config.json"),
            atomically: true,
            encoding: .utf8
        )

        return directory
    }

    private static let sentencePiecePieces: [(token: String, type: Int)] = [
        ("<unk>", 2),
        ("<s>", 3),
        ("</s>", 3),
        ("\u{2581}", 4),
        ("he", 4),
        ("llo", 4),
        ("\u{2581}hello", 4),
        ("\u{2581}there", 4),
        ("he", 4),
        ("<0x21>", 6),
        ("<|im_start|>", 4),
        ("<|im_end|>", 4)
    ]

    private static let sentencePieceTokenizerConfig = [
        #"{"add_bos_token":true,"add_eos_token":false,"bos_token":"<s>","#,
        #""eos_token":"</s>","unk_token":"<unk>","chat_template":"internlm3-test","#,
        #""added_tokens_decoder":{"10":{"content":"<|im_start|>"},"#,
        #""11":{"content":"<|im_end|>"}}}"#
    ].joined()

    private static func makeSentencePieceModel(pieces: [(token: String, type: Int)]) -> Data {
        pieces.reduce(into: Data()) { model, piece in
            var pieceMessage = Data()
            appendFieldBytes(1, Data(piece.token.utf8), to: &pieceMessage)
            appendFieldVarint(3, UInt64(piece.type), to: &pieceMessage)
            appendFieldBytes(1, pieceMessage, to: &model)
        }
    }

    private static func appendFieldBytes(_ fieldNumber: Int, _ value: Data, to data: inout Data) {
        appendVarint(UInt64(fieldNumber << 3 | 2), to: &data)
        appendVarint(UInt64(value.count), to: &data)
        data.append(value)
    }

    private static func appendFieldVarint(_ fieldNumber: Int, _ value: UInt64, to data: inout Data) {
        appendVarint(UInt64(fieldNumber << 3), to: &data)
        appendVarint(value, to: &data)
    }

    private static func appendVarint(_ value: UInt64, to data: inout Data) {
        var value = value
        while value >= 0x80 {
            data.append(UInt8(value & 0x7F) | 0x80)
            value >>= 7
        }
        data.append(UInt8(value))
    }
}
