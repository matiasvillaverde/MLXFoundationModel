import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Jamba architecture")
struct JambaArchitectureTests {
    @Test("decodes Jamba configuration defaults and auto rank")
    func decodesConfigurationDefaultsAndAutoRank() throws {
        let config = try Self.decodeConfig(
            mambaRank: "\"auto\"",
            hiddenLayers: 4,
            numExperts: 3,
            expertLayerOffset: 1,
            expertLayerPeriod: 2
        )

        #expect(config.modelType == "jamba")
        #expect(config.mambaDtRank == 1)
        #expect(config.layersBlockType == ["mamba", "attention", "mamba", "attention"])
        #expect(!config.usesSparseExperts(layerIndex: 0))
        #expect(config.usesSparseExperts(layerIndex: 1))
        #expect(!config.usesSparseExperts(layerIndex: 2))
        #expect(config.usesSparseExperts(layerIndex: 3))
        #expect(config.tieWordEmbeddings)
    }

    @Test("constructs cache, adapters, and scheduled sparse layers")
    func constructsCacheAdaptersAndScheduledSparseLayers() throws {
        let config = try Self.decodeConfig(
            hiddenLayers: 4,
            numExperts: 3,
            expertLayerOffset: 1,
            expertLayerPeriod: 2
        )
        let model = JambaModel(config)
        let layers = model.layers.compactMap { $0 as? JambaDecoderLayer }
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()

        #expect(model.modelType == "jamba")
        #expect(model.kvHeads == [2, 2, 2, 2])
        #expect(layers.map(\.isAttn) == [false, true, false, true])
        #expect(layers.map(\.isSparseMoe) == [false, true, false, true])
        #expect(cache.count == 4)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is KVCacheSimple)
        #expect(loraTargets.count == 2)
    }

    @Test("tiny hybrid model produces finite logits with cache")
    func tinyHybridModelProducesFiniteLogitsWithCache() throws {
        let config = try Self.decodeConfig(hiddenLayers: 2, numExperts: 1)

        Device.withDefaultDevice(.cpu) {
            let model = JambaModel(config)
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(all(isFinite(prefill)).item(Bool.self))
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer normalizes tied head, convolution, and expert tensors")
    func sanitizerNormalizesTiedHeadConvolutionAndExpertTensors() throws {
        let config = try Self.decodeConfig(hiddenLayers: 2, numExperts: 2)
        let model = JambaModel(config)
        let sanitized = model.sanitize(weights: Self.sanitizerFixtureWeights())

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.layers.0.mamba.conv1d.weight"]?.shape == [32, 2, 1])
        #expect(sanitized["model.layers.1.feed_forward.experts.0.gate_proj.weight"] == nil)
        #expect(
            sanitized["model.layers.1.feed_forward.switch_mlp.gate_proj.weight"]?.shape
                == [2, 32, 16]
        )
        #expect(
            sanitized["model.layers.1.feed_forward.switch_mlp.down_proj.weight"]?.shape
                == [2, 16, 32]
        )
        #expect(
            sanitized["model.layers.1.feed_forward.switch_mlp.up_proj.weight"]?.shape
                == [2, 32, 16]
        )
    }

    private static func decodeConfig(
        mambaRank: String = "1",
        hiddenLayers: Int = 2,
        numExperts: Int = 1,
        expertLayerOffset: Int = 0,
        expertLayerPeriod: Int = 1
    ) throws -> JambaConfiguration {
        try JSONDecoder.json5().decode(
            JambaConfiguration.self,
            from: Data(
                Self.configJSON(
                    mambaRank: mambaRank,
                    hiddenLayers: hiddenLayers,
                    numExperts: numExperts,
                    expertLayerOffset: expertLayerOffset,
                    expertLayerPeriod: expertLayerPeriod
                ).utf8
            )
        )
    }

    private static func sanitizerFixtureWeights() -> [String: MLXArray] {
        var weights = [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.layers.0.mamba.conv1d.weight": MLXArray.ones([32, 1, 2])
        ]
        for expert in 0 ..< 2 {
            weights["model.layers.1.block_sparse_moe.experts.\(expert).w1.weight"] =
                MLXArray.ones([32, 16])
            weights["model.layers.1.block_sparse_moe.experts.\(expert).w2.weight"] =
                MLXArray.ones([16, 32])
            weights["model.layers.1.block_sparse_moe.experts.\(expert).w3.weight"] =
                MLXArray.ones([32, 16])
        }
        return weights
    }

    private static func configJSON(
        mambaRank: String,
        hiddenLayers: Int,
        numExperts: Int,
        expertLayerOffset: Int,
        expertLayerPeriod: Int
    ) -> String {
        """
        {
            "model_type": "jamba",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": \(hiddenLayers),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "attn_layer_offset": 1,
            "attn_layer_period": 2,
            "expert_layer_offset": \(expertLayerOffset),
            "expert_layer_period": \(expertLayerPeriod),
            "mamba_d_conv": 2,
            "mamba_d_state": 4,
            "mamba_expand": 2,
            "mamba_dt_rank": \(mambaRank),
            "num_experts": \(numExperts),
            "num_experts_per_tok": 1,
            "rms_norm_eps": 0.00001,
            "max_position_embeddings": 64,
            "vocab_size": 64,
            "tie_word_embeddings": true
        }
        """
    }
}
