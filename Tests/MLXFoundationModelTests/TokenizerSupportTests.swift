import Hub
@testable import MLXLocalModels
import Testing

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
}
