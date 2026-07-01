import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GLM architecture")
struct GLMArchitectureTests {
    @Test("decodes GLM Edge configuration")
    func decodesGLMEdgeConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            GLMConfiguration.self,
            from: Data(Self.glmEdgeConfigJSON.utf8)
        )

        #expect(config.modelType == "glm")
        #expect(config.hiddenSize == 2_048)
        #expect(config.hiddenLayers == 28)
        #expect(config.intermediateSize == 6_144)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 4)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.vocabularySize == 59_264)
        #expect(config.maxPositionEmbeddings == 8_192)
        #expect(!config.attentionBias)
        #expect(config.ropeTheta == 10_000)
        #expect(config.tieWordEmbeddings)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let layout = GLMAttentionLayout(Self.smallConfig(attentionHeads: 4, kvHeads: 2, headDim: 4))

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
        let model = GLMModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
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
            let model = GLMModel(Self.smallConfig())
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
        let model = GLMModel(Self.smallConfig(tieWordEmbeddings: true))
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
        tieWordEmbeddings: Bool = true
    ) -> GLMConfiguration {
        GLMConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: attentionHeads,
            vocabularySize: 64,
            headDim: headDim,
            kvHeads: kvHeads,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let glmEdgeConfigJSON = """
    {
        "architectures": [
            "GlmForCausalLM"
        ],
        "partial_rotary_factor": 1.0,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "eos_token_id": [
            59246,
            59253,
            59255
        ],
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 8192,
        "model_type": "glm",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "pad_token_id": 59246,
        "rms_norm_eps": 1e-05,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.47.0.dev0",
        "use_cache": true,
        "vocab_size": 59264
    }
    """
}
