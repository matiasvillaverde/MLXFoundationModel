import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Mixtral architecture")
struct MixtralArchitectureTests {
    @Test("decodes Mixtral configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            MixtralConfiguration.self,
            from: Data(Self.realConfigJSON.utf8)
        )

        #expect(config.modelType == "mixtral")
        #expect(config.vocabularySize == 32_002)
        #expect(config.hiddenSize == 4_096)
        #expect(config.intermediateSize == 14_336)
        #expect(config.hiddenLayers == 32)
        #expect(config.attentionHeads == 32)
        #expect(config.keyValueHeads == 8)
        #expect(config.expertCount == 8)
        #expect(config.expertsPerToken == 2)
        #expect(config.ropeTheta == 1_000_000)
        #expect(!config.ropeTraditional)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention and routing plans")
    func buildsAttentionAndRoutingPlans() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )
        let layout = MixtralAttentionLayout(config)
        let routing = MixtralRoutingPlan(config)

        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.headDimensions == 4)
        #expect(layout.attentionScale == 0.5)
        #expect(config.ropeScale == 0.25)
        #expect(routing.expertCount == 4)
        #expect(routing.selectedExpertCount == 2)
    }

    @Test("routes top-k experts and normalizes selected logits")
    func routesTopKExpertsAndNormalizesSelectedLogits() {
        Device.withDefaultDevice(.cpu) {
            let routing = MixtralRoutingPlan(Self.smallConfig())
            let logits = MLXArray([Float(0), 5, 1, 4]).reshaped(1, 1, 4)

            let routed = routing.route(logits)
            eval(routed.indices, routed.scores)

            let selected = Set(routed.indices.asArray(Int32.self).map(Int.init))
            let scores = routed.scores.asArray(Float.self)
            #expect(selected == [1, 3])
            #expect(abs(scores.reduce(0, +) - 1) < 0.0001)
        }
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = MixtralModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny Mixtral model produces finite logits")
    func tinyMixtralModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = MixtralModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = MixtralModel(Self.smallConfig())
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
            let model = MixtralModel(Self.smallConfig(hiddenLayers: 1, tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil)

            let gate = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]
            )
            let scales = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.scales"]
            )
            let down = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"]
            )
            let upProjection = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"]
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
        hiddenLayers: Int = 1,
        ropeScaling: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = false
    ) -> MixtralConfiguration {
        MixtralConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 8,
            attentionHeads: 4,
            keyValueHeads: 2,
            expertCount: 4,
            expertsPerToken: 2,
            rmsNormEps: 1e-6,
            vocabularySize: 64,
            ropeTheta: 10_000,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2])
        ]
        let projections = [
            ("w1", Float(1)),
            ("w2", Float(3)),
            ("w3", Float(5))
        ]

        for expertIndex in 0 ..< 4 {
            for (name, baseValue) in projections {
                let value = baseValue + Float(expertIndex)
                weights["model.layers.0.block_sparse_moe.experts.\(expertIndex).\(name).weight"] =
                    Self.filledArray(shape: [2, 2], value: value)
                weights["model.layers.0.block_sparse_moe.experts.\(expertIndex).\(name).scales"] =
                    Self.filledArray(shape: [1], value: value)
                weights["model.layers.0.block_sparse_moe.experts.\(expertIndex).\(name).biases"] =
                    Self.filledArray(shape: [1], value: value)
            }
        }

        return weights
    }

    private static func filledArray(shape: [Int], value: Float) -> MLXArray {
        MLXArray([Float](repeating: value, count: shape.reduce(1, *))).reshaped(shape)
    }

    private static var realConfigJSON: String {
        #"""
        {
            "model_type": "mixtral",
            "vocab_size": 32002,
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_local_experts": 8,
            "num_experts_per_tok": 2,
            "rms_norm_eps": 0.00001,
            "rope_theta": 1000000.0,
            "tie_word_embeddings": false
        }
        """#
    }
}
