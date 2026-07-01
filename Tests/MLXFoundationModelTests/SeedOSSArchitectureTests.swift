import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Seed OSS architecture")
struct SeedOSSArchitectureTests {
    @Test("decodes Seed OSS configuration with default RoPE metadata")
    func decodesConfigurationWithDefaultRoPEMetadata() throws {
        let config = try JSONDecoder.json5().decode(
            SeedOSSConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "seed_oss")
        #expect(config.hiddenSize == 5_120)
        #expect(config.hiddenLayers == 64)
        #expect(config.intermediateSize == 27_648)
        #expect(config.attentionHeads == 80)
        #expect(config.kvHeads == 8)
        #expect(config.resolvedHeadDimensions == 128)
        #expect(config.ropeTheta == 10_000_000)
        #expect(config.maxPositionEmbeddings == 524_288)
        #expect(config.ropeScaling?["rope_type"] == .string("default"))
        #expect(!config.ropeTraditional)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds grouped-query attention layout")
    func buildsGroupedQueryAttentionLayout() {
        let layout = SeedOSSAttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = SeedOSSModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = SeedOSSModel(Self.smallConfig(hiddenLayers: 2))
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

    @Test("tiny tied model produces finite logits")
    func tinyTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = SeedOSSModel(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer strips rotary sidecars and tied output head tensors")
    func sanitizerStripsRotarySidecarsAndTiedOutputHeadTensors() {
        let model = SeedOSSModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([1]),
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
        tieWordEmbeddings: Bool = false
    ) -> SeedOSSConfiguration {
        SeedOSSConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            headDimensions: 4,
            maxPositionEmbeddings: 64,
            ropeTheta: 10_000,
            ropeScaling: ["rope_type": .string("default")],
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let configJSON = """
        {
            "model_type": "seed_oss",
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "intermediate_size": 27648,
            "num_attention_heads": 80,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.000001,
            "vocab_size": 155136,
            "rope_theta": 10000000.0,
            "rope_scaling": {
                "rope_type": "default"
            },
            "max_position_embeddings": 524288,
            "tie_word_embeddings": false
        }
        """
}
