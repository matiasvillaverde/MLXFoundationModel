import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("OLMo3 architecture")
struct Olmo3ArchitectureTests {
    @Test("decodes OLMo3 configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Olmo3Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 4)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 4)
        #expect(config.maxPositionEmbeddings == 65_536)
        #expect(config.slidingWindow == 4_096)
        #expect(config.ropeTheta == 500_000)
        #expect(config.attentionBias == false)
        #expect(config.tieWordEmbeddings == false)
        #expect(config.layerTypes == [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ])
    }

    @Test("builds OLMo3 layer schedule")
    func buildsLayerSchedule() {
        let schedule = Olmo3LayerSchedule(types: [
            "sliding_attention",
            "sliding_attention",
            "full_attention"
        ])

        #expect(schedule.firstSlidingIndex == 0)
        #expect(schedule.firstFullIndex == 2)
        #expect(schedule.usesFullAttention(layerIndex: 0) == false)
        #expect(schedule.usesFullAttention(layerIndex: 2) == true)
    }

    @Test("builds OLMo3 attention layout")
    func buildsAttentionLayout() {
        let sliding = Olmo3AttentionLayout(Self.smallConfig(), layerType: "sliding_attention")
        let full = Olmo3AttentionLayout(Self.smallConfig(), layerType: "full_attention")

        #expect(sliding.hiddenSize == 16)
        #expect(sliding.queryHeads == 4)
        #expect(sliding.keyValueHeads == 2)
        #expect(sliding.headSize == 4)
        #expect(sliding.queryProjectionSize == 16)
        #expect(sliding.keyValueProjectionSize == 8)
        #expect(sliding.usesFullAttention == false)
        #expect(full.usesFullAttention == true)
    }

    @Test("constructs OLMo3 model with caches, greedy fast path, and LoRA layers")
    func constructsModelWithCachesGreedyFastPathAndLoRALayers() throws {
        let model = Olmo3Model(Self.smallConfig(layerTypes: [
            "sliding_attention",
            "full_attention"
        ]))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("removes stale rotary inverse-frequency weights")
    func removesStaleRotaryInverseFrequencyWeights() {
        let model = Olmo3Model(Self.smallConfig())
        let weights = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1]),
            "model.embed_tokens.weight": MLXArray([2])
        ]

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    @Test("runs a small OLMo3 forward pass")
    func runsSmallForwardPass() {
        Device.withDefaultDevice(.cpu) {
            let model = Olmo3Model(Self.smallConfig(layerTypes: ["full_attention"]))
            let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])

            let logits = model(tokens, cache: nil)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    private static func smallConfig(
        layerTypes: [String] = ["sliding_attention"]
    ) -> Olmo3Configuration {
        Olmo3Configuration(
            hiddenSize: 16,
            hiddenLayers: layerTypes.count,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 128,
            slidingWindow: 16,
            ropeTheta: 10_000,
            layerTypes: layerTypes
        )
    }
}
