import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MiniMax architecture")
struct MiniMaxArchitectureTests {
    @Test("decodes MiniMax configuration with project defaults")
    func decodesMiniMaxConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "num_experts_per_tok": 1,
            "num_local_experts": 2,
            "num_hidden_layers": 1,
            "rms_norm_eps": 0.00001,
            "rope_theta": 10000,
            "rotary_dim": 4,
            "vocab_size": 32
        }
        """#

        let config = try JSONDecoder.json5().decode(
            MiniMaxConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "minimax")
        #expect(config.sharedIntermediateSize == 16)
        #expect(config.tieWordEmbeddings == false)
        #expect(config.scoringFunc == "sigmoid")
        #expect(config.useQkNorm == true)
    }

    @Test("builds attention layout and routing plan")
    func buildsAttentionLayoutAndRoutingPlan() {
        let config = Self.smallConfig(headDim: 4)
        let layout = MiniMaxAttentionLayout(config)
        let routingPlan = MiniMaxRoutingPlan(config)

        #expect(layout.hiddenSize == 8)
        #expect(layout.attentionHeads == 2)
        #expect(layout.keyValueHeads == 1)
        #expect(layout.headDimensions == 4)
        #expect(layout.queryProjectionSize == 8)
        #expect(layout.keyValueProjectionSize == 4)
        #expect(layout.attentionScale == 0.5)
        #expect(routingPlan.expertCount == 2)
        #expect(routingPlan.selectedExpertCount == 1)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = MiniMaxModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "minimax")
        #expect(model.vocabularySize == 32)
        #expect(model.kvHeads == [1, 1])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny MiniMax model produces finite logits")
    func tinyMiniMaxModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = MiniMaxModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 32])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs per-expert checkpoint weights")
    func sanitizerPacksPerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = MiniMaxModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.block_sparse_moe.experts.0.w1.weight"] == nil)

            let gate = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"]
            )
            let down = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"]
            )
            let upProjection = try #require(
                sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"]
            )

            eval(gate, down, upProjection)
            #expect(gate.shape == [2, 2, 2])
            #expect(down.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [6, 6, 6, 6])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        headDim: Int? = nil,
        tieWordEmbeddings: Bool = false
    ) -> MiniMaxConfiguration {
        MiniMaxConfiguration(
            hiddenSize: 8,
            intermediateSize: 16,
            attentionHeads: 2,
            kvHeads: 1,
            maxPositionEmbeddings: 64,
            numExpertsPerTok: 1,
            numLocalExperts: 2,
            hiddenLayers: hiddenLayers,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000,
            rotaryDim: 4,
            vocabularySize: 32,
            tieWordEmbeddings: tieWordEmbeddings,
            headDim: headDim
        )
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights = ["lm_head.weight": MLXArray.ones([2, 2])]
        let projections = [("w1", Float(1)), ("w2", Float(3)), ("w3", Float(5))]
        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in projections {
                weights[
                    "model.layers.0.block_sparse_moe.experts.\(expertIndex).\(name).weight"
                ] = MLXArray(
                    [Float](repeating: baseValue + Float(expertIndex), count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }
}
