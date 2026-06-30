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
}
