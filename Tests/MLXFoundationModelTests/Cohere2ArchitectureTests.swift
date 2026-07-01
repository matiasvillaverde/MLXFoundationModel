import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Cohere2 architecture")
struct Cohere2ArchitectureTests {
    @Test("decodes tiny Aya Global checkpoint configuration")
    func decodesTinyAyaGlobalCheckpointConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            Cohere2Configuration.self,
            from: Data(Self.tinyAyaGlobalConfigJSON.utf8)
        )

        #expect(config.modelType == "cohere2")
        #expect(config.hiddenSize == 2_048)
        #expect(config.hiddenLayers == 4)
        #expect(config.intermediateSize == 11_008)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 4)
        #expect(config.headDimensions == 128)
        #expect(config.ropeTheta == 50_000)
        #expect(config.vocabularySize == 262_144)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.logitScale == 1)
        #expect(config.slidingWindow == 4_096)
        #expect(config.slidingWindowPattern == 4)
        #expect(config.layerTypes == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ])
    }

    @Test("builds layer schedule")
    func buildsLayerSchedule() {
        let schedule = Cohere2LayerSchedule(Self.smallConfig(hiddenLayers: 5))

        #expect(schedule.layerCount == 5)
        #expect(schedule.slidingWindowPattern == 4)
        #expect(schedule.firstSlidingLayerIndex == 0)
        #expect(schedule.firstFullLayerIndex == 3)
        #expect(schedule.usesSlidingWindow(layerIndex: 0))
        #expect(schedule.usesSlidingWindow(layerIndex: 2))
        #expect(!schedule.usesSlidingWindow(layerIndex: 3))
    }

    @Test("builds attention layout")
    func buildsAttentionLayout() {
        let sliding = Cohere2AttentionLayout(Self.smallConfig(), usesSlidingWindow: true)
        let full = Cohere2AttentionLayout(Self.smallConfig(), usesSlidingWindow: false)

        #expect(sliding.hiddenSize == 16)
        #expect(sliding.queryHeads == 4)
        #expect(sliding.keyValueHeads == 2)
        #expect(sliding.headSize == 4)
        #expect(sliding.queryProjectionSize == 16)
        #expect(sliding.keyValueProjectionSize == 8)
        #expect(sliding.attentionScale == 0.5)
        #expect(sliding.usesSlidingWindow)
        #expect(!full.usesSlidingWindow)
    }

    @Test("constructs model with mixed caches and greedy fast path")
    func constructsModelWithMixedCachesAndGreedyFastPath() throws {
        let model = Cohere2Model(Self.smallConfig(hiddenLayers: 4))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2, 2, 2])
        #expect(cache.count == 4)
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? RotatingKVCache)
        _ = try #require(cache[2] as? RotatingKVCache)
        _ = try #require(cache[3] as? KVCacheSimple)
        #expect(loraTargets.count == 4)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Cohere2Model(Self.smallConfig(hiddenLayers: 1))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer removes stale rotary inverse-frequency weights")
    func sanitizerRemovesStaleRotaryInverseFrequencyWeights() {
        let model = Cohere2Model(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(hiddenLayers: Int = 2) -> Cohere2Configuration {
        Cohere2Configuration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            kvHeads: 2,
            headDimensions: 4,
            vocabularySize: 64,
            logitScale: 1,
            slidingWindow: 8,
            slidingWindowPattern: 4
        )
    }

    private static let tinyAyaGlobalConfigJSON = #"""
    {
        "model_type": "cohere2",
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "rope_theta": 50000,
        "vocab_size": 262144,
        "layer_norm_eps": 0.00001,
        "logit_scale": 1.0,
        "attention_bias": false,
        "sliding_window": 4096,
        "sliding_window_pattern": 4,
        "layer_types": [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ]
    }
    """#
}
