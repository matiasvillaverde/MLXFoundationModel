import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Granite MoE architecture")
struct GraniteMoEArchitectureTests {
    @Test("decodes Granite MoE configuration with checkpoint defaults")
    func decodesGraniteMoEConfigurationWithDefaults() throws {
        let configuration = try Self.realCheckpointConfiguration()

        #expect(configuration.modelType == "granitemoe")
        #expect(configuration.hiddenSize == 1_024)
        #expect(configuration.hiddenLayers == 24)
        #expect(configuration.intermediateSize == 512)
        #expect(configuration.attentionHeads == 16)
        #expect(configuration.kvHeads == 8)
        #expect(configuration.localExperts == 32)
        #expect(configuration.expertsPerToken == 8)
        #expect(configuration.logitsScaling == 6)
        #expect(configuration.attentionMultiplier == 1.0 / 64.0)
        #expect(configuration.embeddingMultiplier == 12)
        #expect(configuration.residualMultiplier == 0.22)
        #expect(configuration.ropeTheta == 1_500_000)
        #expect(configuration.tieWordEmbeddings)
    }

    @Test("builds attention, RoPE, and routing plans")
    func buildsPlans() {
        let configuration = Self.smallConfiguration(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )
        let attention = GraniteMoEAttentionLayout(configuration)
        let rope = GraniteMoERoPEPlan(configuration, dimensions: attention.headSize)
        let routing = GraniteMoERoutingPlan(configuration)

        #expect(attention.queryProjectionSize == 16)
        #expect(attention.keyValueProjectionSize == 8)
        #expect(attention.headSize == 4)
        #expect(attention.attentionScale == 0.25)
        #expect(rope.dimensions == 4)
        #expect(rope.base == 10_000)
        #expect(rope.scale == 0.25)
        #expect(routing.expertCount == 4)
        #expect(routing.selectedExpertCount == 2)
    }

    @Test("routes over selected raw logits")
    func routesOverSelectedRawLogits() {
        Device.withDefaultDevice(.cpu) {
            let routing = GraniteMoERoutingPlan(Self.smallConfiguration())
            let logits = MLXArray([Float(0), 5, 1, 4]).reshaped(1, 1, 4)

            let route = routing.route(logits)
            eval(route.indices, route.gates)

            let selected = Set(route.indices.asArray(Int32.self).map(Int.init))
            let gates = route.gates.asArray(Float.self)
            #expect(selected == [1, 3])
            #expect(abs(gates.reduce(0, +) - 1) < 0.0001)
        }
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = GraniteMoEModel(Self.smallConfiguration(hiddenLayers: 2))
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
            let model = GraniteMoEModel(Self.smallConfiguration())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer remaps packed block sparse MoE weights")
    func sanitizerRemapsPackedBlockSparseMoEWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GraniteMoEModel(
                Self.smallConfiguration(hiddenLayers: 1, tieWordEmbeddings: true)
            )
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.0.block_sparse_moe.input_linear.weight"] == nil)
            #expect(sanitized["model.layers.0.block_sparse_moe.output_linear.weight"] == nil)

            let gate = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]
            )
            let upProjection = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"]
            )
            let down = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"]
            )

            eval(gate, upProjection, down)
            #expect(gate.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(down.shape == [2, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [2, 2, 2, 2])
        }
    }

    private static func smallConfiguration(
        hiddenLayers: Int = 1,
        ropeScaling: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = true
    ) -> GraniteMoEConfiguration {
        GraniteMoEConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 8,
            attentionHeads: 4,
            rmsNormEps: 1e-6,
            vocabularySize: 64,
            logitsScaling: 2,
            attentionMultiplier: 0.25,
            embeddingMultiplier: 1,
            residualMultiplier: 0.5,
            maxPositionEmbeddings: 128,
            kvHeads: 2,
            attentionBias: false,
            ropeTheta: 10_000,
            localExperts: 4,
            expertsPerToken: 2,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func realCheckpointConfiguration() throws -> GraniteMoEConfiguration {
        let json = #"""
        {
            "model_type": "granitemoe",
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "intermediate_size": 512,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_local_experts": 32,
            "num_experts_per_tok": 8,
            "rms_norm_eps": 0.000001,
            "vocab_size": 49155
        }
        """#

        return try JSONDecoder.json5().decode(
            GraniteMoEConfiguration.self,
            from: Data(json.utf8)
        )
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.0.block_sparse_moe.input_linear.weight": concatenated(
                [
                    MLXArray.ones([2, 2, 2]),
                    Self.filledArray(shape: [2, 2, 2], value: 2)
                ],
                axis: 1
            ),
            "model.layers.0.block_sparse_moe.output_linear.weight": Self.filledArray(
                shape: [2, 2, 2],
                value: 3
            )
        ]
    }

    private static func filledArray(shape: [Int], value: Float) -> MLXArray {
        MLXArray([Float](repeating: value, count: shape.reduce(1, *))).reshaped(shape)
    }
}
