import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("OLMoE architecture")
struct OlmoEArchitectureTests {
    @Test("decodes OLMoE configuration with project defaults")
    func decodesOlmoEConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "num_experts": 2,
            "num_experts_per_tok": 1
        }
        """#

        let config = try JSONDecoder.json5().decode(
            OlmoEConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "olmoe")
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.ropeTraditional == false)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.attentionBias == false)
        #expect(config.mlpBias == false)
        #expect(config.normTopkProb == false)
    }

    @Test("builds OLMoE attention layout and routing plan")
    func buildsOlmoEAttentionLayoutAndRoutingPlan() {
        let config = Self.smallConfig(headDimensions: 4)
        let layout = OlmoEAttentionLayout(config)
        let routingPlan = OlmoERoutingPlan(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDimensions == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
        #expect(routingPlan.expertCount == 2)
        #expect(routingPlan.selectedExpertCount == 1)
        #expect(routingPlan.normalizesSelectedProbabilities == false)
    }

    @Test("router normalizes selected probabilities when configured")
    func routerNormalizesSelectedProbabilitiesWhenConfigured() {
        let routingPlan = OlmoERoutingPlan(
            Self.smallConfig(numExperts: 3, numExpertsPerToken: 2, normTopkProb: true)
        )
        let routed = routingPlan.route(MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3))

        eval(routed.scores)

        #expect(routed.indices.shape == [1, 1, 2])
        #expect(abs(routed.scores.asArray(Float.self).reduce(0, +) - 1) < 0.0001)
    }

    @Test("constructs OLMoE model with cache, adapters, and greedy fast path")
    func constructsOlmoEModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = OlmoEModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "olmoe")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny OLMoE model produces finite logits")
    func tinyOlmoEModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = OlmoEModel(Self.smallConfig(tieWordEmbeddings: false))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs per-expert checkpoint weights")
    func sanitizerPacksPerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = OlmoEModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.up_proj.weight"] == nil)

            let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
            let down = try #require(sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
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
        headDimensions: Int? = nil,
        tieWordEmbeddings: Bool = true,
        numExperts: Int = 2,
        numExpertsPerToken: Int = 1,
        normTopkProb: Bool = false
    ) -> OlmoEConfiguration {
        OlmoEConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            headDimensions: headDimensions,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings,
            numExperts: numExperts,
            numExpertsPerToken: numExpertsPerToken,
            normTopkProb: normTopkProb
        )
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights = ["lm_head.weight": MLXArray.ones([2, 2])]
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(3)),
            ("up_proj", Float(5))
        ]

        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in projections {
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: baseValue + Float(expertIndex), count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }
}
