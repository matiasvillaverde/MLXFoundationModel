import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Gemma2 architecture")
struct Gemma2ArchitectureTests {
    @Test("decodes Gemma2 configuration with project defaults")
    func decodesGemma2ConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Gemma2Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.headDimensions == 4)
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.attnLogitSoftcapping == 50)
        #expect(config.finalLogitSoftcapping == 30)
        #expect(config.queryPreAttnScalar == 4)
    }

    @Test("builds grouped Gemma2 attention layout")
    func buildsGroupedGemma2AttentionLayout() {
        let config = Self.smallConfig()

        let layout = Gemma2AttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.keyValueGroups == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.25)
        #expect(layout.attentionLogitSoftCap == 25)
    }

    @Test("decodes explicit Gemma2 softcap fields")
    func decodesExplicitGemma2SoftcapFields() throws {
        let json = #"""
        {
            "hidden_size": 24,
            "num_hidden_layers": 1,
            "intermediate_size": 48,
            "num_attention_heads": 4,
            "head_dim": 8,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "attn_logit_softcapping": 40,
            "final_logit_softcapping": 20,
            "query_pre_attn_scalar": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Gemma2Configuration.self,
            from: Data(json.utf8)
        )
        let layout = Gemma2AttentionLayout(config)

        #expect(config.headDimensions == 8)
        #expect(config.kvHeads == 2)
        #expect(config.finalLogitSoftcapping == 20)
        #expect(layout.attentionScale == 0.125)
        #expect(layout.attentionLogitSoftCap == 40)
    }

    @Test("constructs Gemma2 model with greedy fast path and LoRA layers")
    func constructsGemma2ModelWithGreedyFastPathAndLoRALayers() {
        let config = Self.smallConfig()

        let model = Gemma2Model(config)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    private static func smallConfig() -> Gemma2Configuration {
        Gemma2Configuration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            headDimensions: 4,
            vocabularySize: 64,
            kvHeads: 2,
            attnLogitSoftcapping: 25,
            finalLogitSoftcapping: 15,
            queryPreAttnScalar: 16
        )
    }
}
