import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Helium architecture")
struct HeliumArchitectureTests {
    @Test("decodes Helium configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            HeliumConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "helium")
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.resolvedHeadDim == 4)
        #expect(config.maxPositionEmbeddings == 4_096)
        #expect(config.ropeTheta == 100_000)
        #expect(config.rmsNormEps == 1e-8)
        #expect(!config.attentionBias)
        #expect(!config.mlpBias)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let layout = HeliumAttentionLayout(
            Self.smallConfig(attentionHeads: 4, kvHeads: 2, headDim: 4)
        )

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDim == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = HeliumModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits with cache")
    func tinyModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = HeliumModel(Self.smallConfig())
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(prefill)).item(Bool.self))
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer strips tied head and rotary checkpoint tensors")
    func sanitizerStripsTiedHeadAndRotaryCheckpointTensors() {
        let model = HeliumModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        attentionHeads: Int = 4,
        kvHeads: Int? = nil,
        headDim: Int? = nil,
        tieWordEmbeddings: Bool = false
    ) -> HeliumConfiguration {
        HeliumConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            vocabularySize: 64,
            ropeTheta: 100_000,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "helium",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            \(includeOptionalFields ? "\"num_key_value_heads\": 2," : "")
            \(includeOptionalFields ? "\"head_dim\": 4," : "")
            "vocab_size": 64
        }
        """
    }
}
