import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Hunyuan V1 Dense architecture")
struct HunyuanV1DenseArchitectureTests {
    @Test("decodes Hunyuan MT configuration")
    func decodesHunyuanMTConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            HunyuanV1DenseConfiguration.self,
            from: Data(Self.hunyuanMTConfigJSON.utf8)
        )

        #expect(config.modelType == "hunyuan_v1_dense")
        #expect(config.vocabularySize == 128_256)
        #expect(config.hiddenSize == 4_096)
        #expect(config.hiddenLayers == 32)
        #expect(config.intermediateSize == 14_336)
        #expect(config.attentionHeads == 32)
        #expect(config.kvHeads == 8)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.ropeTheta == 10_000)
        #expect(config.ropeAlpha == 100_000)
        #expect(config.maxPositionEmbeddings == 32_768)
        #expect(config.useQKNorm)
        #expect(config.tieWordEmbeddings)
    }

    @Test("builds dynamic-alpha RoPE plan")
    func buildsDynamicAlphaRoPEPlan() {
        let plan = HunyuanV1DenseRoPEPlan(dimensions: 128, base: 10_000, alpha: 100_000)

        #expect(plan.dimensions == 128)
        #expect(plan.base == 10_000)
        #expect(plan.alpha == 100_000)
        let expectedBase = Float(10_000) * pow(Float(100_000), Float(128) / Float(126))
        #expect(plan.adjustedBase == expectedBase)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let layout = HunyuanV1DenseAttentionLayout(Self.smallConfig())

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
        let model = HunyuanV1DenseModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = HunyuanV1DenseModel(Self.smallConfig())
            let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(tokens, cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

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
        let model = HunyuanV1DenseModel(Self.smallConfig(tieWordEmbeddings: true))
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
        tieWordEmbeddings: Bool = true
    ) -> HunyuanV1DenseConfiguration {
        HunyuanV1DenseConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            kvHeads: 2,
            ropeScaling: [
                "alpha": .float(1),
                "factor": .float(1),
                "type": .string("dynamic")
            ],
            tieWordEmbeddings: tieWordEmbeddings,
            headDim: 4
        )
    }

    private static let hunyuanMTConfigJSON = """
    {
        "add_classification_head": false,
        "architectures": [
            "HunYuanDenseV1ForCausalLM"
        ],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attention_head_dim": 128,
        "bos_token_id": 1,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "max_position_embeddings": 32768,
        "mlp_bias": false,
        "model_type": "hunyuan_v1_dense",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pad_token_id": 0,
        "rms_norm_eps": 1e-05,
        "rope_scaling": {
            "alpha": 100000.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "type": "dynamic"
        },
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.41.2",
        "use_cache": true,
        "use_qk_norm": true,
        "use_rotary_pos_emb": true,
        "vocab_size": 128256
    }
    """
}
