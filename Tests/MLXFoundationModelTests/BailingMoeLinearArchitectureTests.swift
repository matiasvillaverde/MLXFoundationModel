import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Bailing MoE Linear architecture")
struct BailingMoeLinearArchitectureTests {
    @Test("decodes Bailing MoE Linear configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            BailingMoeLinearConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "bailing_moe_linear")
        #expect(config.hiddenSize == 16)
        #expect(config.intermediateSize == 32)
        #expect(config.moeIntermediateSize == 8)
        #expect(config.numExperts == 2)
        #expect(config.numSharedExperts == 1)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 2)
        #expect(config.resolvedHeadDim == 4)
        #expect(config.layerGroupSize == 2)
        #expect(config.groupNormSize == 2)
        #expect(config.ropeTraditional)
        #expect(config.useQKNorm)
        #expect(config.moeRouterEnableSharedExpert)
    }

    @Test("builds global and linear layer plan")
    func buildsGlobalAndLinearLayerPlan() {
        let plan = BailingMoeLinearLayerPlan(Self.smallConfig())

        #expect(plan.isGlobal == [false, true, false, true])
        #expect(plan.firstLinearLayerIndex == 0)
        #expect(plan.firstGlobalLayerIndex == 1)
        #expect(plan.usesSparseExperts(Self.smallConfig(), layerIndex: 0) == false)
        #expect(plan.usesSparseExperts(Self.smallConfig(), layerIndex: 1))
    }

    @Test("registers and constructs Bailing MoE Linear through the factory")
    func registersAndConstructsThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("bailing_moe_linear"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BailingMoeLinearArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "bailing_moe_linear"
        )

        #expect(model is BailingMoeLinearModel)
        #expect((model as? BailingMoeLinearModel)?.vocabularySize == 64)
    }

    @Test("constructs mixed caches, adapters, and greedy fast path")
    func constructsMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = BailingMoeLinearModel(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "bailing_moe_linear")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [4, 2, 4, 2])
        #expect(cache.count == 4)
        _ = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        _ = try #require(cache[2] as? MambaCache)
        _ = try #require(cache[3] as? KVCacheSimple)
        #expect(loraTargets.count == 4)
        #expect(loraTargets[0].1 == ["query_key_value"])
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = BailingMoeLinearModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache.map(\.offset) == [3, 3, 3, 3])
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny tied model uses embedding head")
    func tinyTiedModelUsesEmbeddingHead() {
        Device.withDefaultDevice(.cpu) {
            let model = BailingMoeLinearModel(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs experts, remaps gate, and normalizes head")
    func sanitizerPacksExpertsRemapsGateAndNormalizesHead() throws {
        let model = BailingMoeLinearModel(Self.smallConfig(normHead: true))
        let sanitized = model.sanitize(weights: Self.checkpointWeights())

        #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.gate.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.attention.rotary_emb.inv_freq"] == nil)

        let head = try #require(sanitized["lm_head.weight"])
        let values = head.asArray(Float.self)
        #expect(values.allSatisfy { abs($0 - 0.70710677) < 0.001 })
    }

    private static func smallConfig(
        normHead: Bool = false,
        tieWordEmbeddings: Bool = false
    ) -> BailingMoeLinearConfiguration {
        BailingMoeLinearConfiguration(
            hiddenSize: 16,
            intermediateSize: 32,
            maxPositionEmbeddings: 128,
            moeIntermediateSize: 8,
            numExperts: 2,
            numSharedExperts: 1,
            normTopkProb: true,
            attentionHeads: 4,
            numExpertsPerToken: 1,
            hiddenLayers: 4,
            kvHeads: 2,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            firstKDenseReplace: 1,
            layerGroupSize: 2,
            groupNormSize: 2,
            ropeTraditional: true,
            normHead: normHead,
            useQKNorm: true,
            tieWordEmbeddings: tieWordEmbeddings,
            moeRouterEnableExpertBias: true,
            routedScalingFactor: 2,
            scoreFunction: "sigmoid",
            headDim: 4
        )
    }

    private static let configJSON = """
    {
        "model_type": "bailing_moe_linear",
        "hidden_size": 16,
        "intermediate_size": 32,
        "max_position_embeddings": 128,
        "moe_intermediate_size": 8,
        "num_experts": 2,
        "num_shared_experts": 1,
        "norm_topk_prob": true,
        "num_attention_heads": 4,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "vocab_size": 64,
        "first_k_dense_replace": 1,
        "layer_group_size": 2,
        "group_norm_size": 2,
        "rope_traditional": true,
        "use_qkv_bias": false,
        "use_qk_norm": true,
        "partial_rotary_factor": 1.0,
        "moe_router_enable_expert_bias": true,
        "routed_scaling_factor": 2,
        "score_function": "sigmoid",
        "n_group": 1,
        "topk_group": 1,
        "moe_router_enable_shared_expert": true,
        "head_dim": 4
    }
    """

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.1.mlp.gate.weight": MLXArray.ones([2, 2]),
            "model.layers.1.attention.rotary_emb.inv_freq": MLXArray.ones([1])
        ]
        for expertIndex in 0 ..< 2 {
            weights["model.layers.1.mlp.experts.\(expertIndex).gate_proj.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 1)
            weights["model.layers.1.mlp.experts.\(expertIndex).down_proj.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 3)
            weights["model.layers.1.mlp.experts.\(expertIndex).up_proj.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 5)
        }
        return weights
    }
}
