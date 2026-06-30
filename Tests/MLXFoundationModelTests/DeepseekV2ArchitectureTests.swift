import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("DeepSeek V2 architecture")
struct DeepseekV2ArchitectureTests {
    @Test("decodes DeepSeek V2 routing configuration")
    func decodesDeepseekV2RoutingConfiguration() throws {
        let json = #"""
        {
            "model_type": "deepseek_v2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "kv_lora_rank": 6,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "q_lora_rank": null,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "vocab_size": 64,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "group_limited_greedy",
            "scoring_func": "softmax"
        }
        """#

        let config = try JSONDecoder.json5().decode(
            DeepseekV2Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "deepseek_v2")
        #expect(config.qLoraRank == nil)
        #expect(config.topKMethod == .groupLimitedGreedy)
        #expect(config.routingScoreFunction == .softmax)
        #expect(config.usesExpertScoreCorrectionBias == false)
    }

    @Test("plans DeepSeek V2 grouped softmax routing")
    func plansDeepseekV2GroupedSoftmaxRouting() {
        let plan = DeepseekV3RoutingPlan(Self.smallConfig())

        #expect(plan.routedExperts == 4)
        #expect(plan.expertsPerToken == 2)
        #expect(plan.groupCount == 2)
        #expect(plan.keptGroupCount == 1)
        #expect(plan.topKMethod == .groupLimitedGreedy)
        #expect(plan.routingScoreFunction == .softmax)
        #expect(plan.normalizeTopK == false)
    }

    @Test("registers DeepSeek V2 model type")
    func registersDeepseekV2ModelType() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("deepseek_v2"))
    }

    @Test("constructs DeepSeek V2 model through the factory")
    func constructsDeepseekV2ModelThroughFactory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("DeepseekV2ArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "deepseek_v2"
        )

        #expect(model is DeepseekV2Model)
        #expect((model as? DeepseekV2Model)?.vocabularySize == 64)
    }

    @Test("tiny DeepSeek V2 model produces finite logits")
    func tinyDeepseekV2ModelProducesFiniteLogits() {
        let model = DeepseekV2Model(Self.smallConfig())
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        let model = DeepseekV2Model(Self.smallConfig())
        let output = model.greedyToken(
            LMInput.Text(tokens: MLXArray([1, 2, 3])),
            cache: nil,
            state: nil
        )
        eval(output.token)

        #expect(output.token.shape == [1])
    }

    @Test("sanitizer packs DeepSeek V2 expert weights")
    func sanitizerPacksDeepseekV2ExpertWeights() {
        let model = DeepseekV2Model(Self.smallConfig())
        var weights: [String: MLXArray] = [:]

        for expert in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.1.mlp.experts.\(expert).up_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.1.mlp.experts.\(expert).down_proj.weight"] = MLXArray.ones([16, 8])
        }

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 8, 16])
        #expect(sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 8, 16])
        #expect(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape == [4, 16, 8])
        #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
    }

    private static var configJSON: String {
        #"""
        {
            "model_type": "deepseek_v2",
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "kv_lora_rank": 6,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "q_lora_rank": null,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "vocab_size": 64,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "n_group": 2,
            "topk_group": 1,
            "topk_method": "group_limited_greedy",
            "scoring_func": "softmax",
            "first_k_dense_replace": 1,
            "max_position_embeddings": 64
        }
        """#
    }

    private static func smallConfig() -> DeepseekV2Configuration {
        DeepseekV2Configuration(
            modelType: "deepseek_v2",
            vocabSize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            nSharedExperts: 1,
            nRoutedExperts: 4,
            kvLoraRank: 6,
            qLoraRank: nil,
            qkRopeHeadDim: 4,
            vHeadDim: 4,
            qkNopeHeadDim: 4,
            topKMethod: .groupLimitedGreedy,
            routingScoreFunction: .softmax,
            nGroup: 2,
            topkGroup: 1,
            numExpertsPerTok: 2,
            firstKDenseReplace: 1,
            maxPositionEmbeddings: 64
        )
    }
}
