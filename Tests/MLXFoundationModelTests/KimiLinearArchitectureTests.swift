import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Kimi Linear architecture")
struct KimiLinearArchitectureTests {
    @Test("decodes Kimi Linear configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            KimiLinearConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "kimi_linear")
        #expect(config.vocabularySize == 64)
        #expect(config.hiddenSize == 16)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 2)
        #expect(config.linearAttention.numHeads == 4)
        #expect(config.linearAttention.headDim == 4)
        #expect(config.linearAttention.kdaLayers == [1, 3])
        #expect(config.linearAttention.shortConvKernelSize == 3)
        #expect(config.qkNopeHeadDim == 4)
        #expect(config.qkRopeHeadDim == 4)
        #expect(config.valueHeadDim == 4)
        #expect(config.numExpertsPerToken == 1)
    }

    @Test("builds KDA and MLA layer plan")
    func buildsLayerPlan() {
        let config = Self.smallConfig()
        let plan = KimiLinearLayerPlan(config)

        #expect(plan.isLinear == [true, false, true, false])
        #expect(plan.firstLinearLayerIndex == 0)
        #expect(plan.firstAttentionLayerIndex == 1)
        #expect(plan.usesSparseExperts(config, layerIndex: 0) == false)
        #expect(plan.usesSparseExperts(config, layerIndex: 1))
        #expect(plan.usesSparseExperts(config, layerIndex: 2))
    }

    @Test("registers and constructs Kimi Linear through the factory")
    func registersAndConstructsThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("kimi_linear"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("KimiLinearArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "kimi_linear"
        )

        #expect(model is KimiLinearModel)
        #expect((model as? KimiLinearModel)?.vocabularySize == 64)
    }

    @Test("constructs mixed caches, adapters, and greedy fast path")
    func constructsMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = KimiLinearModel(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "kimi_linear")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [4, 4, 4, 4])
        #expect(cache.count == 4)
        let firstCache = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        _ = try #require(cache[2] as? MambaCache)
        _ = try #require(cache[3] as? KVCacheSimple)
        #expect(firstCache.state.count == 4)
        #expect(loraTargets.count == 4)
        #expect(loraTargets[0].1.contains("q_proj"))
        #expect(loraTargets[1].1.contains("kv_b_proj"))
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = KimiLinearModel(Self.smallConfig())
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
            let model = KimiLinearModel(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs experts, remaps KDA weights, and splits MLA projection")
    func sanitizerPacksExpertsRemapsKDAWeightsAndSplitsMLAProjection() throws {
        let model = KimiLinearModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: Self.checkpointWeights())

        #expect(sanitized["model.mtp.block.weight"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.1.block_sparse_moe.experts.0.w1.weight"] == nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.gate.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.e_score_correction_bias"] != nil)

        let conv = try #require(sanitized["model.layers.0.self_attn.q_conv.conv.weight"])
        #expect(conv.shape == [16, 3, 1])
        #expect(sanitized["model.layers.0.self_attn.dt_bias"]?.shape == [16])
        #expect(sanitized["model.layers.0.self_attn.A_log"]?.shape == [4])
        #expect(sanitized["model.layers.1.self_attn.kv_b_proj.weight"] == nil)
        #expect(sanitized["model.layers.1.self_attn.embed_q.weight"]?.shape == [4, 4, 4])
        #expect(sanitized["model.layers.1.self_attn.unembed_out.weight"]?.shape == [4, 4, 4])
    }

    private static func smallConfig(tieWordEmbeddings: Bool = false) -> KimiLinearConfiguration {
        KimiLinearConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: 4,
            attentionHeads: 4,
            kvHeads: 2,
            intermediateSize: 32,
            headDim: 8,
            ropeTheta: 100,
            rmsNormEps: 1e-6,
            linearAttention: KimiLinearAttentionConfiguration(
                numHeads: 4,
                headDim: 4,
                kdaLayers: [1, 3],
                shortConvKernelSize: 3
            ),
            modelMaxLength: 128,
            numExperts: 2,
            moeIntermediateSize: 8,
            kvLoraRank: 4,
            tieWordEmbeddings: tieWordEmbeddings,
            qkNopeHeadDim: 4,
            qkRopeHeadDim: 4,
            valueHeadDim: 4,
            numExpertsPerToken: 1,
            numSharedExperts: 1,
            firstKDenseReplace: 1
        )
    }

    private static let configJSON = """
    {
        "model_type": "kimi_linear",
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "head_dim": 8,
        "rope_theta": 100,
        "rms_norm_eps": 1e-6,
        "linear_attn_config": {
            "num_heads": 4,
            "head_dim": 4,
            "kda_layers": [1, 3],
            "short_conv_kernel_size": 3
        },
        "model_max_length": 128,
        "num_experts": 2,
        "moe_intermediate_size": 8,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "num_experts_per_tok": 1,
        "num_shared_experts": 1,
        "first_k_dense_replace": 1
    }
    """

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "model.mtp.block.weight": MLXArray.ones([1]),
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.0.self_attn.q_conv1d.weight": MLXArray.ones([16, 1, 3]),
            "model.layers.0.self_attn.dt_bias": MLXArray.ones([1, 16]),
            "model.layers.0.self_attn.A_log": MLXArray.ones([1, 1, 4, 1]),
            "model.layers.1.self_attn.kv_b_proj.weight": MLXArray.ones([32, 4]),
            "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.1.block_sparse_moe.gate.weight": MLXArray.ones([2, 2]),
            "model.layers.1.block_sparse_moe.gate.e_score_correction_bias": MLXArray.ones([2])
        ]
        for expertIndex in 0 ..< 2 {
            weights["model.layers.1.block_sparse_moe.experts.\(expertIndex).w1.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 1)
            weights["model.layers.1.block_sparse_moe.experts.\(expertIndex).w2.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 3)
            weights["model.layers.1.block_sparse_moe.experts.\(expertIndex).w3.weight"] =
                MLXArray.ones([2, 2]) * Float(expertIndex + 5)
        }
        return weights
    }
}
