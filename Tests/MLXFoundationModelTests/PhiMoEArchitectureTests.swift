import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Phi MoE architecture")
struct PhiMoEArchitectureTests {
    @Test("decodes Phi MoE configuration with project defaults")
    func decodesPhiMoEConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 8,
            "num_attention_heads": 4,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            PhiMoEConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "phimoe")
        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 8)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 8)
        #expect(config.maxPositionEmbeddings == 131_072)
        #expect(config.originalMaxPositionEmbeddings == 4_096)
        #expect(config.numLocalExperts == 16)
        #expect(config.numExpertsPerToken == 2)
    }

    @Test("builds Phi MoE attention and router plans")
    func buildsPhiMoEAttentionAndRouterPlans() {
        let config = Self.smallConfig()

        let layout = PhiMoEAttentionLayout(config)
        let router = PhiMoERouterPlan(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
        #expect(router.expertCount == 4)
        #expect(router.expertsPerToken == 2)
    }

    @Test("plans Phi MoE linear rotary embeddings")
    func plansPhiMoELinearRotaryEmbeddings() {
        let linearConfig = Self.smallConfig(
            ropeScaling: Phi3RoPEScaling(
                longFactor: nil,
                shortFactor: nil,
                factor: 4,
                type: "linear",
                longMScale: nil,
                shortMScale: nil
            )
        )
        let linearPlan = PhiMoERotaryPlan(
            linearConfig,
            layout: PhiMoEAttentionLayout(linearConfig)
        )

        #expect(linearPlan.kind == .rope(scale: 0.25))
    }

    @Test("plans Phi MoE LongRoPE rotary embeddings")
    func plansPhiMoELongRoPERotaryEmbeddings() {
        let longConfig = Self.smallConfig(
            ropeScaling: Phi3RoPEScaling(
                longFactor: [1, 2],
                shortFactor: [1, 1],
                factor: nil,
                type: "su",
                longMScale: 1.25,
                shortMScale: 0.75
            ),
            maxPositionEmbeddings: 128,
            originalMaxPositionEmbeddings: 64
        )
        let longPlan = PhiMoERotaryPlan(
            longConfig,
            layout: PhiMoEAttentionLayout(longConfig)
        )

        #expect(longPlan.maxPositionEmbeddings == 128)
        #expect(longPlan.originalMaxPositionEmbeddings == 64)
        #expect(
            longPlan.kind == .longRoPE(
                shortFactor: [1, 1],
                longFactor: [1, 2],
                shortMScale: 0.75,
                longMScale: 1.25
            )
        )
    }

    @Test("constructs Phi MoE model with cache and greedy fast path")
    func constructsPhiMoEModelWithCacheAndGreedyFastPath() {
        let model = PhiMoEModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny Phi MoE model produces finite logits")
    func tinyPhiMoEModelProducesFiniteLogits() {
        let model = PhiMoEModel(Self.smallConfig(numLocalExperts: 2, numExpertsPerToken: 1))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer packs complete experts and preserves incomplete expert tensors")
    func sanitizerPacksCompleteExpertsAndPreservesIncompleteTensors() {
        let model = PhiMoEModel(Self.smallConfig(hiddenLayers: 2, numLocalExperts: 2))
        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2])
        ]

        for expert in 0 ..< 2 {
            weights["model.layers.0.block_sparse_moe.experts.\(expert).w1.weight"] =
                MLXArray.ones([8, 16])
            weights["model.layers.0.block_sparse_moe.experts.\(expert).w2.weight"] =
                MLXArray.ones([16, 8])
            weights["model.layers.0.block_sparse_moe.experts.\(expert).w3.weight"] =
                MLXArray.ones([8, 16])
        }
        weights["model.layers.1.block_sparse_moe.experts.0.w1.weight"] =
            MLXArray.ones([8, 16])

        let sanitized = model.sanitize(weights: weights)

        #expect(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]?.shape
                == [2, 8, 16]
        )
        #expect(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"]?.shape
                == [2, 16, 8]
        )
        #expect(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"]?.shape
                == [2, 8, 16]
        )
        #expect(sanitized["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.1.block_sparse_moe.experts.0.w1.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        ropeScaling: Phi3RoPEScaling? = nil,
        maxPositionEmbeddings: Int = 64,
        originalMaxPositionEmbeddings: Int = 64,
        numLocalExperts: Int = 4,
        numExpertsPerToken: Int = 2
    ) -> PhiMoEConfiguration {
        PhiMoEConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 8,
            hiddenLayers: hiddenLayers,
            attentionHeads: 4,
            kvHeads: 2,
            maxPositionEmbeddings: maxPositionEmbeddings,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            ropeScaling: ropeScaling,
            numLocalExperts: numLocalExperts,
            numExpertsPerToken: numExpertsPerToken
        )
    }
}
