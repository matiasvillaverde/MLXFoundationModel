import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Phixtral architecture")
struct PhixtralArchitectureTests {
    @Test("decodes Phi-MSFT checkpoint configuration")
    func decodesPhiMSFTCheckpointConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            PhixtralConfiguration.self,
            from: Data(Self.phixtral2x28ConfigJSON.utf8)
        )

        #expect(config.modelType == "phi-msft")
        #expect(config.vocabularySize == 51_200)
        #expect(config.hiddenSize == 2_560)
        #expect(config.attentionHeads == 32)
        #expect(config.hiddenLayers == 32)
        #expect(config.rotaryDimensions == 32)
        #expect(config.intermediateSize == 10_240)
        #expect(config.numExpertsPerToken == 2)
        #expect(config.numLocalExperts == 2)
        #expect(config.layerNormEps == 1e-5)
    }

    @Test("builds attention and router plans")
    func buildsAttentionAndRouterPlans() {
        let config = Self.smallConfig()
        let attention = PhixtralAttentionLayout(config)
        let router = PhixtralRouterPlan(config)

        #expect(attention.hiddenSize == 16)
        #expect(attention.attentionHeads == 4)
        #expect(attention.headSize == 4)
        #expect(attention.rotaryDimensions == 4)
        #expect(attention.attentionScale == 0.5)
        #expect(router.expertCount == 2)
        #expect(router.expertsPerToken == 1)
    }

    @Test("constructs model with cache and greedy fast path")
    func constructsModelWithCacheAndGreedyFastPath() {
        let model = PhixtralModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [4, 4])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["Wqkv", "out_proj"])
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        let model = PhixtralModel(Self.smallConfig())
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer packs complete expert MLP tensors")
    func sanitizerPacksCompleteExpertMLPTensors() {
        let model = PhixtralModel(Self.smallConfig(hiddenLayers: 2, numLocalExperts: 2))
        var weights: [String: MLXArray] = [:]

        for expertIndex in 0 ..< 2 {
            weights["transformer.h.0.moe.mlp.\(expertIndex).fc1.weight"] =
                MLXArray.ones([32, 16])
            weights["transformer.h.0.moe.mlp.\(expertIndex).fc1.bias"] =
                MLXArray.ones([32])
            weights["transformer.h.0.moe.mlp.\(expertIndex).fc2.weight"] =
                MLXArray.ones([16, 32])
        }
        weights["transformer.h.1.moe.mlp.0.fc1.weight"] = MLXArray.ones([32, 16])

        let sanitized = model.sanitize(weights: weights)

        #expect(
            sanitized["transformer.h.0.moe.switch_mlp.fc1.weight"]?.shape
                == [2, 32, 16]
        )
        #expect(
            sanitized["transformer.h.0.moe.switch_mlp.fc1.bias"]?.shape
                == [2, 32]
        )
        #expect(
            sanitized["transformer.h.0.moe.switch_mlp.fc2.weight"]?.shape
                == [2, 16, 32]
        )
        #expect(sanitized["transformer.h.0.moe.mlp.0.fc1.weight"] == nil)
        #expect(sanitized["transformer.h.1.moe.mlp.0.fc1.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        numLocalExperts: Int = 2,
        numExpertsPerToken: Int = 1
    ) -> PhixtralConfiguration {
        PhixtralConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            attentionHeads: 4,
            hiddenLayers: hiddenLayers,
            rotaryDimensions: 4,
            intermediateSize: 32,
            numExpertsPerToken: numExpertsPerToken,
            numLocalExperts: numLocalExperts
        )
    }

    private static let phixtral2x28ConfigJSON = #"""
    {
        "model_type": "phi-msft",
        "vocab_size": 51200,
        "n_embd": 2560,
        "n_head": 32,
        "n_layer": 32,
        "n_inner": null,
        "rotary_dim": 32,
        "num_experts_per_tok": 2,
        "num_local_experts": 2,
        "layer_norm_epsilon": 0.00001
    }
    """#
}
