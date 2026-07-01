import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Step3.5 architecture")
struct Step3p5ArchitectureTests {
    @Test("decodes configuration and per-layer settings")
    func decodesConfigurationAndPerLayerSettings() throws {
        let config = try JSONDecoder.json5().decode(
            Step3p5Configuration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "step3p5")
        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 3)
        #expect(config.vocabularySize == 64)
        #expect(config.attentionHeads == 4)
        #expect(config.attentionGroups == 2)
        #expect(config.headDim == 4)
        #expect(config.layerType(at: 0) == "sliding_attention")
        #expect(config.layerType(at: 1) == "full_attention")
        #expect(config.ropeTheta(at: 2) == 30_000)
        #expect(config.partialRotaryFactor(at: 1) == 0.5)
        #expect(config.swigluLimit(at: 1) == 4)
        #expect(config.swigluLimit(at: 2) == nil)
        #expect(config.sharedSwigluLimit(at: 0) == 3)
        #expect(config.yarnOnlyTypes == ["full_attention"])
        #expect(config.ropeScaling(at: 0) == nil)
        #expect(config.ropeScaling(at: 1)?["type"] == .string("yarn"))
        #expect(config.attentionOverride?.attentionHeads == 2)
        #expect(config.attentionOverride?.attentionGroups == 1)
    }

    @Test("builds layer and cache plan")
    func buildsLayerAndCachePlan() {
        let plan = Step3p5LayerPlan(Self.smallConfig())

        #expect(plan.layerTypes == ["sliding_attention", "full_attention", "sliding_attention"])
        #expect(plan.firstSlidingLayerIndex == 0)
        #expect(plan.firstFullLayerIndex == 1)
        #expect(plan.isMoELayer(0) == false)
        #expect(plan.isMoELayer(1))
        #expect(plan.isMoELayer(2))
    }

    @Test("registers and constructs Step3.5 through the factory")
    func registersAndConstructsStep3p5ThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("step3p5"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("Step3p5ArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.configJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "step3p5"
        )

        #expect(model is Step3p5Model)
        #expect((model as? Step3p5Model)?.vocabularySize == 64)
    }

    @Test("constructs mixed caches, adapters, and greedy fast path")
    func constructsMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = Step3p5Model(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "step3p5")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [1, 2, 1])
        #expect(cache.count == 3)
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        _ = try #require(cache[2] as? RotatingKVCache)
        #expect(loraTargets.count == 6)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = Step3p5Model(Self.smallConfig())
            let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(tokens, cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache.map(\.offset) == [3, 3, 3])
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny tied model uses embedding head")
    func tinyTiedModelUsesEmbeddingHead() {
        Device.withDefaultDevice(.cpu) {
            let model = Step3p5Model(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer remaps vanilla MoE weights and drops sidecars")
    func sanitizerRemapsVanillaMoEWeightsAndDropsSidecars() throws {
        let model = Step3p5Model(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "model.layers.1.moe.gate_proj.weight": MLXArray.ones([4, 16, 16]),
            "model.layers.1.moe.router_bias": MLXArray.ones([4]),
            "model.layers.1.share_expert.gate_proj.weight": MLXArray.ones([16, 16]),
            "model.layers.10.self_attn.q_proj.weight": MLXArray.ones([16, 16]),
            "model.mtp.layers.0.weight": MLXArray.ones([1]),
            "model.layers.0.input_layernorm.weight": MLXArray.ones([16]),
            "lm_head.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.1.mlp.gate.router_bias"] != nil)
        #expect(sanitized["model.layers.1.mlp.share_expert.gate_proj.weight"] != nil)
        #expect(sanitized["model.layers.10.self_attn.q_proj.weight"] == nil)
        #expect(sanitized["model.mtp.layers.0.weight"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)

        let norm = try #require(sanitized["model.layers.0.input_layernorm.weight"])
        #expect(norm.asArray(Float.self).allSatisfy { abs($0 - 2) < 0.0001 })
    }

    private static func smallConfig(tieWordEmbeddings: Bool = false) -> Step3p5Configuration {
        Step3p5Configuration(
            hiddenSize: 16,
            hiddenLayers: 3,
            vocabularySize: 64,
            attentionHeads: 4,
            attentionGroups: 2,
            headDim: 4,
            intermediateSize: 32,
            rmsNormEps: 1e-5,
            ropeTheta: .float(10_000),
            slidingWindow: 4,
            layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
            partialRotaryFactors: [1, 1, 1],
            attentionOverride: Step3p5AttentionOverride(
                attentionHeads: 2,
                attentionGroups: 1
            ),
            usesHeadWiseAttentionGate: true,
            expertCount: 4,
            expertsPerToken: 2,
            moeIntermediateSize: 16,
            sharedExpertIntermediateSize: 16,
            moeLayers: "1,2",
            routerScaling: 2,
            swigluLimits: [nil, 4, nil],
            sharedSwigluLimits: [3, nil, 5],
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let configJSON = """
    {
        "model_type": "step3p5",
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "vocab_size": 64,
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-5,
        "rope_theta": [10000, 20000, 30000],
        "rope_scaling": {
            "type": "yarn",
            "factor": 2,
            "original_max_position_embeddings": 4096
        },
        "max_position_embeddings": 8192,
        "sliding_window": 4,
        "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
        "yarn_only_types": ["full_attention"],
        "partial_rotary_factors": [1.0, 0.5, 1.0],
        "attention_other_setting": {
            "num_attention_heads": 2,
            "num_attention_groups": 1
        },
        "use_head_wise_attn_gate": true,
        "moe_num_experts": 4,
        "moe_top_k": 2,
        "moe_intermediate_size": 16,
        "share_expert_dim": 16,
        "moe_layers_enum": "1,2",
        "moe_router_scaling_factor": 2,
        "norm_expert_weight": true,
        "swiglu_limits": [null, 4.0, 0.0],
        "swiglu_limits_shared": [3.0, null, 5.0],
        "tie_word_embeddings": false
    }
    """
}
