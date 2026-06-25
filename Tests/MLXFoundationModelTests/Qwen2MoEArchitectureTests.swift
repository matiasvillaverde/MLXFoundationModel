import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Qwen2 MoE architecture")
struct Qwen2MoEArchitectureTests {
    @Test("decodes Qwen2 MoE configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            Qwen2MoEConfiguration.self,
            from: Data(Self.configJSON(includeKVHeads: false).utf8)
        )

        #expect(config.modelType == "qwen2_moe")
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.ropeTheta == 1_000_000)
        #expect(!config.ropeTraditional)
        #expect(!config.tieWordEmbeddings)
        #expect(!config.normTopkProb)
    }

    @Test("builds attention and routing plans")
    func buildsAttentionAndRoutingPlans() {
        let config = Self.smallConfig()
        let layout = Qwen2MoEAttentionLayout(config)
        let routing = Qwen2MoERoutingPlan(config)

        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.headDimensions == 4)
        #expect(layout.attentionScale == 0.5)
        #expect(routing.expertCount == 2)
        #expect(routing.selectedExpertCount == 1)
        #expect(!routing.normalizesSelectedProbabilities)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = Qwen2MoEModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Qwen2MoEModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs quantized experts and strips unused tensors")
    func sanitizerPacksQuantizedExpertsAndStripsUnusedTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen2MoEModel(Self.smallConfig(hiddenLayers: 1, tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
            #expect(sanitized["model.layers.0.mlp.shared_expert.gate_proj.weight"] != nil)
            #expect(sanitized["model.layers.0.mlp.shared_expert_gate.weight"] != nil)

            let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
            let gateScales = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.scales"])
            let down = try #require(sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"])

            eval(gate, gateScales, down, upProjection)
            #expect(gate.shape == [2, 2, 2])
            #expect(gateScales.shape == [2, 1])
            #expect(down.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [6, 6, 6, 6])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        tieWordEmbeddings: Bool = false
    ) -> Qwen2MoEConfiguration {
        Qwen2MoEConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            numExpertsPerToken: 1,
            numExperts: 2,
            moeIntermediateSize: 8,
            sharedExpertIntermediateSize: 12,
            rmsNormEps: 1e-6,
            vocabularySize: 64,
            kvHeads: 2,
            ropeTheta: 10_000,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.0.mlp.shared_expert.down_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.0.mlp.shared_expert.up_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.0.mlp.shared_expert_gate.weight": MLXArray.ones([1, 2])
        ]
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(3)),
            ("up_proj", Float(5))
        ]

        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in projections {
                let value = baseValue + Float(expertIndex)
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: value, count: 4)
                )
                .reshaped([2, 2])
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).scales"] = MLXArray(
                    [Float](repeating: value, count: 1)
                )
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).biases"] = MLXArray(
                    [Float](repeating: value, count: 1)
                )
            }
        }

        return weights
    }

    private static func configJSON(includeKVHeads: Bool) -> String {
        """
        {
            "model_type": "qwen2_moe",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            \(includeKVHeads ? "\"num_key_value_heads\": 2," : "")
            "num_experts_per_tok": 1,
            "num_experts": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 12,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64
        }
        """
    }
}
