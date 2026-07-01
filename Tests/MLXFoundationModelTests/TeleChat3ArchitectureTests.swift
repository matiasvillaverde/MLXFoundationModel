import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("TeleChat3 architecture")
struct TeleChat3ArchitectureTests {
    @Test("decodes TeleChat3 configuration with YaRN scaling")
    func decodesConfigurationWithYarnScaling() throws {
        let config = try JSONDecoder.json5().decode(
            TeleChat3Configuration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "telechat3")
        #expect(config.hiddenSize == 6_144)
        #expect(config.intermediateSize == 24_576)
        #expect(config.maxPositionEmbeddings == 32_768)
        #expect(config.attentionHeads == 48)
        #expect(config.hiddenLayers == 64)
        #expect(config.keyValueHeads == 8)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.ropeTheta == 1_000_000)
        #expect(config.ropeScaling?["type"] == .string("telechat3-yarn"))
        #expect(!config.attentionBias)
        #expect(!config.mlpBias)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds grouped attention layout")
    func buildsGroupedAttentionLayout() {
        let layout = TeleChat3AttentionLayout(Self.smallConfig(attentionHeads: 4, kvHeads: 2))

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
        let model = TeleChat3Model(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
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
            let model = TeleChat3Model(Self.smallConfig())
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
        let model = TeleChat3Model(Self.smallConfig(tieWordEmbeddings: true))
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
    ) -> TeleChat3Configuration {
        TeleChat3Configuration(
            hiddenSize: 16,
            intermediateSize: 32,
            attentionHeads: attentionHeads,
            hiddenLayers: hiddenLayers,
            keyValueHeads: kvHeads,
            vocabularySize: 64,
            headDim: headDim,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let configJSON = """
    {
        "model_type": "telechat3",
        "hidden_size": 6144,
        "intermediate_size": 24576,
        "max_position_embeddings": 32768,
        "num_attention_heads": 48,
        "num_hidden_layers": 64,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "vocab_size": 131072,
        "rope_theta": 1000000.0,
        "mlp_bias": false,
        "attention_bias": false,
        "head_dim": 128,
        "rope_scaling": {
            "factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "telechat3-yarn",
            "type": "telechat3-yarn"
        },
        "tie_word_embeddings": false
    }
    """
}
