import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("StableLM architecture")
struct StableLMArchitectureTests {
    @Test("decodes StableLM configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            StableLMConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "stablelm")
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.ropeTheta == 10_000)
        #expect(config.partialRotaryFactor == 0.25)
        #expect(!config.useParallelResidual)
        #expect(!config.qkLayerNorm)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention layout with partial RoPE")
    func buildsAttentionLayoutWithPartialRoPE() {
        let config = Self.smallConfig(attentionHeads: 4, kvHeads: 2, partialRotaryFactor: 0.5)
        let layout = StableLMAttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDimensions == 4)
        #expect(layout.rotaryDimensions == 2)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = StableLMModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny sequential model produces finite logits")
    func tinySequentialModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = StableLMModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny parallel per-head normalized model produces finite logits with cache")
    func tinyParallelPerHeadNormalizedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = StableLMModel(
                Self.smallConfig(
                    hiddenLayers: 2,
                    kvHeads: 2,
                    useQKVBias: false,
                    useParallelResidual: true,
                    qkLayerNorm: true
                )
            )
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

    @Test("sanitizer strips tied head and rotary checkpoint tensors")
    func sanitizerStripsTiedHeadAndRotaryCheckpointTensors() {
        let model = StableLMModel(Self.smallConfig(tieWordEmbeddings: true))
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
        partialRotaryFactor: Float = 0.5,
        useQKVBias: Bool = true,
        useParallelResidual: Bool = false,
        qkLayerNorm: Bool = false,
        tieWordEmbeddings: Bool = false
    ) -> StableLMConfiguration {
        StableLMConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            maxPositionEmbeddings: 64,
            useQKVBias: useQKVBias,
            partialRotaryFactor: partialRotaryFactor,
            useParallelResidual: useParallelResidual,
            qkLayerNorm: qkLayerNorm,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "stablelm",
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            \(includeOptionalFields ? "\"num_key_value_heads\": 2," : "")
            \(includeOptionalFields ? "\"use_parallel_residual\": true," : "")
            \(includeOptionalFields ? "\"qk_layernorm\": true," : "")
            "layer_norm_eps": 0.00001,
            "use_qkv_bias": true
        }
        """
    }
}
