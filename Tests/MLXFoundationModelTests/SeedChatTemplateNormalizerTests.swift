import Hub
@testable import MLXLocalModels
import Testing

@Suite("Seed chat template normalizer")
struct SeedChatTemplateNormalizerTests {
    @Test("normalizes numeric-key budget table")
    func normalizesNumericKeyBudgetTable() throws {
        let rewriter = TokenizerConfigurationRewriter(registry: TokenizerReplacementRegistry())
        let config = Config([
            "tokenizer_class": Config("PreTrainedTokenizer"),
            "chat_template": Config(Self.seedChatTemplateWithNumericKeys)
        ])

        let rewritten = try #require(rewriter.rewrite(config).dictionary()?["chat_template"]?.string())

        #expect(rewritten.contains("{%- set budget_reflections_v05 = ["))
        #expect(rewritten.contains("[512, 128],"))
        #expect(rewritten.contains("[16384, 1024]"))
        #expect(rewritten.contains("{%- for k, v in budget_reflections_v05 -%}"))
        #expect(rewritten.contains("{%- set ns.interval = 1024 -%}"))
        #expect(!rewritten.contains("512:"))
        #expect(!rewritten.contains("16384:"))
        #expect(!rewritten.contains("| dictsort"))
    }

    @Test("leaves ordinary chat templates unchanged")
    func leavesOrdinaryChatTemplatesUnchanged() {
        let rewriter = TokenizerConfigurationRewriter(registry: TokenizerReplacementRegistry())
        let config = Config([
            "chat_template": Config("{{ bos_token }}{{ messages[0]['content'] }}")
        ])

        #expect(rewriter.rewrite(config) == config)
    }

    private static let seedChatTemplateWithNumericKeys = """
        {%- set budget_reflections_v05 = {
            0:      0,
            512:    128,
            16384:  1024
        } -%}
        {%- for k, v in budget_reflections_v05 | dictsort -%}
        {{ k }}={{ v }}
        {%- endfor -%}
        {%- set ns.interval = budget_reflections_v05[16384] -%}
        """
}
