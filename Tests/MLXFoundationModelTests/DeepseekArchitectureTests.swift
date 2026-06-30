import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("DeepSeek architecture")
struct DeepseekArchitectureTests {
    @Test("decodes DeepSeek MoE configuration")
    func decodesDeepseekMoEConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            DeepseekConfiguration.self,
            from: Data(Self.realMoEConfigJSON.utf8)
        )

        #expect(config.modelType == "deepseek")
        #expect(config.vocabularySize == 102_400)
        #expect(config.hiddenSize == 2_048)
        #expect(config.intermediateSize == 10_944)
        #expect(config.moeIntermediateSize == 1_408)
        #expect(config.hiddenLayers == 28)
        #expect(config.attentionHeads == 16)
        #expect(config.keyValueHeads == 16)
        #expect(config.sharedExperts == 2)
        #expect(config.routedExperts == 64)
        #expect(config.expertsPerToken == 6)
        #expect(config.firstDenseReplacementLayer == 1)
        #expect(config.scoringFunction == "softmax")
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention, RoPE, layer, and routing plans")
    func buildsPlans() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )
        let attention = DeepseekAttentionLayout(config)
        let rope = DeepseekRoPEPlan(config, dimensions: attention.headDimensions)
        let layers = DeepseekLayerPlan(config)
        let routing = DeepseekRoutingPlan(config)

        #expect(attention.queryProjectionSize == 16)
        #expect(attention.keyValueProjectionSize == 8)
        #expect(attention.headDimensions == 4)
        #expect(attention.attentionScale == 0.5)
        #expect(rope.scale == 0.25)
        #expect(!layers.usesMoE(at: 0))
        #expect(layers.usesMoE(at: 1))
        #expect(layers.usesMoE(at: 2))
        #expect(routing.routedExperts == 4)
        #expect(routing.expertsPerToken == 2)
    }

    @Test("routes top-k experts from softmax probabilities")
    func routesTopKExpertsFromSoftmaxProbabilities() {
        Device.withDefaultDevice(.cpu) {
            let routing = DeepseekRoutingPlan(Self.smallConfig())
            let logits = MLXArray([Float(0), 5, 1, 4]).reshaped(1, 1, 4)

            let routed = routing.route(logits)
            eval(routed.indices, routed.scores)

            let selected = Set(routed.indices.asArray(Int32.self).map(Int.init))
            let scores = routed.scores.asArray(Float.self)
            #expect(selected == [1, 3])
            #expect(scores.reduce(0, +) < 1)
        }
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = DeepseekModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "deepseek")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny DeepSeek MoE model produces finite logits")
    func tinyDeepseekMoEModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = DeepseekModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = DeepseekModel(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("sanitizer packs quantized expert tensors")
    func sanitizerPacksQuantizedExpertTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = DeepseekModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.1.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)

            let gate = try #require(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
            let scales = try #require(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.scales"])
            let down = try #require(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
            )

            eval(gate, scales, down, upProjection)
            #expect(gate.shape == [4, 2, 2])
            #expect(scales.shape == [4, 1])
            #expect(down.shape == [4, 2, 2])
            #expect(upProjection.shape == [4, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [8, 8, 8, 8])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 2,
        ropeScaling: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = false
    ) -> DeepseekConfiguration {
        DeepseekConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            hiddenLayers: hiddenLayers,
            attentionHeads: 4,
            keyValueHeads: 2,
            sharedExperts: 1,
            routedExperts: 4,
            expertsPerToken: 2,
            moeLayerFrequency: 1,
            firstDenseReplacementLayer: 1,
            maxPositionEmbeddings: 64,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.1.self_attn.rotary_emb.inv_freq": MLXArray.ones([2])
        ]
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(3)),
            ("up_proj", Float(5))
        ]

        for expertIndex in 0 ..< 4 {
            for (name, baseValue) in projections {
                let value = baseValue + Float(expertIndex)
                weights["model.layers.1.mlp.experts.\(expertIndex).\(name).weight"] =
                    Self.filledArray(shape: [2, 2], value: value)
                weights["model.layers.1.mlp.experts.\(expertIndex).\(name).scales"] =
                    Self.filledArray(shape: [1], value: value)
                weights["model.layers.1.mlp.experts.\(expertIndex).\(name).biases"] =
                    Self.filledArray(shape: [1], value: value)
            }
        }
        return weights
    }

    private static func filledArray(shape: [Int], value: Float) -> MLXArray {
        MLXArray([Float](repeating: value, count: shape.reduce(1, *))).reshaped(shape)
    }

    private static var realMoEConfigJSON: String {
        #"""
        {
            "model_type": "deepseek",
            "vocab_size": 102400,
            "hidden_size": 2048,
            "intermediate_size": 10944,
            "moe_intermediate_size": 1408,
            "num_hidden_layers": 28,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "n_shared_experts": 2,
            "n_routed_experts": 64,
            "num_experts_per_tok": 6,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 1,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000,
            "scoring_func": "softmax",
            "tie_word_embeddings": false
        }
        """#
    }
}
