import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GPT-BigCode architecture")
struct GPTBigCodeArchitectureTests {
    @Test("decodes GPT-BigCode configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            GPTBigCodeConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "gpt_bigcode")
        #expect(config.multiQuery)
        #expect(config.resolvedKVHeads == 1)
        #expect(config.feedForwardSize == 64)
        #expect(config.attentionBias)
        #expect(config.mlpBias)
        #expect(config.tieWordEmbeddings)
        #expect(config.activationFunction == "gelu_pytorch_tanh")
    }

    @Test("builds multi-query attention layout")
    func buildsMultiQueryAttentionLayout() {
        let layout = GPTBigCodeAttentionLayout(Self.smallConfig(attentionHeads: 4, kvHeads: 1))

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.keyValueHeads == 1)
        #expect(layout.headDimensions == 4)
        #expect(layout.keyValueDimensions == 4)
        #expect(layout.combinedProjectionSize == 24)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("builds full multi-head attention layout")
    func buildsFullMultiHeadAttentionLayout() {
        let layout = GPTBigCodeAttentionLayout(
            Self.smallConfig(attentionHeads: 4, multiQuery: false)
        )

        #expect(layout.keyValueHeads == 4)
        #expect(layout.keyValueDimensions == 16)
        #expect(layout.combinedProjectionSize == 48)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = GPTBigCodeModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 1))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [1, 1])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["c_attn", "c_proj"])
    }

    @Test("tiny tied model produces finite logits with cache")
    func tinyTiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = GPTBigCodeModel(Self.smallConfig(hiddenLayers: 2))
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny untied full attention model produces finite logits")
    func tinyUntiedFullAttentionModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = GPTBigCodeModel(
                Self.smallConfig(multiQuery: false, tieWordEmbeddings: false)
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer preserves raw keys and strips tied head tensors")
    func sanitizerPreservesRawKeysAndStripsTiedHeadTensors() {
        let model = GPTBigCodeModel(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "transformer.wte.weight": MLXArray.ones([64, 16]),
            "transformer.h.0.attn.bias": MLXArray.ones([1]),
            "transformer.h.0.attn.masked_bias": MLXArray.ones([1]),
            "transformer.h.0.attn.c_attn.weight": MLXArray.ones([24, 16]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "model.lm_head.scales": MLXArray.ones([1])
        ])

        #expect(sanitized["transformer.wte.weight"] != nil)
        #expect(sanitized["transformer.h.0.attn.bias"] == nil)
        #expect(sanitized["transformer.h.0.attn.masked_bias"] == nil)
        #expect(sanitized["transformer.h.0.attn.c_attn.weight"] != nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
    }

    @Test("sanitizer normalizes model-prefixed raw keys")
    func sanitizerNormalizesModelPrefixedRawKeys() {
        let model = GPTBigCodeModel(Self.smallConfig(tieWordEmbeddings: false))
        let sanitized = model.sanitize(weights: [
            "model.transformer.h.0.mlp.c_fc.weight": MLXArray.ones([64, 16]),
            "model.lm_head.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["transformer.h.0.mlp.c_fc.weight"] != nil)
        #expect(sanitized["lm_head.weight"] != nil)
    }

    private static func smallConfig(
        attentionHeads: Int = 4,
        hiddenLayers: Int = 1,
        kvHeads: Int? = nil,
        multiQuery: Bool = true,
        tieWordEmbeddings: Bool = true
    ) -> GPTBigCodeConfiguration {
        GPTBigCodeConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 64,
            attentionHeads: attentionHeads,
            maxPositionEmbeddings: 64,
            vocabularySize: 64,
            kvHeads: kvHeads,
            multiQuery: multiQuery,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "gpt_bigcode",
            "n_embd": 16,
            "n_head": 4,
            "n_inner": 64,
            "n_layer": 1,
            "n_positions": 64,
            \(includeOptionalFields ? "\"num_key_value_heads\": 2," : "")
            \(includeOptionalFields ? "\"multi_query\": false," : "")
            "layer_norm_epsilon": 0.00001,
            "vocab_size": 64
        }
        """
    }
}
