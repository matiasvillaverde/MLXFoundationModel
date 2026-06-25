import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("AfMoE architecture")
struct AfMoEArchitectureTests {
    @Test("decodes AfMoE configuration with project defaults")
    func decodesAfMoEConfigurationWithDefaults() throws {
        let json = #"""
        {
            "model_type": "afmoe",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4
        }
        """#

        let config = try JSONDecoder.json5().decode(
            AfMoEConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "afmoe")
        #expect(config.layerTypes == ["full_attention", "full_attention"])
        #expect(config.numExperts == 128)
        #expect(config.numExpertsPerToken == 8)
        #expect(config.routeNorm == true)
        #expect(config.routeScale == 2.826)
        #expect(config.mupEnabled == true)
    }

    @Test("builds full and sliding attention layouts")
    func buildsFullAndSlidingAttentionLayouts() {
        let config = Self.smallConfig(layerTypes: ["full_attention", "sliding_attention"])
        let full = AfMoEAttentionLayout(config, kind: .full)
        let sliding = AfMoEAttentionLayout(config, kind: .sliding)
        let layers = AfMoELayerPlan(config)

        #expect(full.attentionHeads == 4)
        #expect(full.keyValueHeads == 2)
        #expect(full.queryProjectionSize == 16)
        #expect(full.usesRotaryPosition == false)
        #expect(sliding.usesRotaryPosition == true)
        #expect(layers.firstFullLayer == 0)
        #expect(layers.firstSlidingLayer == 1)
        #expect(layers.usesSparseExperts(layerIndex: 0) == false)
        #expect(layers.usesSparseExperts(layerIndex: 1) == true)
    }

    @Test("router uses expert bias for selection only")
    func routerUsesExpertBiasForSelectionOnly() {
        let routing = AfMoERoutingPlan(
            Self.smallConfig(numExperts: 3, routeScale: 2)
        )
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)
        let bias = MLXArray([Float(0), Float(10), Float(0)])
        let routed = routing.route(logits: logits, expertBias: bias, outputDType: .float32)

        eval(routed.scores, routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [1])
        #expect(abs(routed.scores.item(Float.self) - (sigmoid(1) * 2)) < 0.0001)
    }

    @Test("router masks lower-scoring groups before expert selection")
    func routerMasksLowerScoringGroupsBeforeExpertSelection() {
        let routing = AfMoERoutingPlan(
            Self.smallConfig(numExperts: 4, nGroup: 2, topkGroup: 1)
        )
        let logits = MLXArray([Float(1), Float(4), Float(3), Float(2)]).reshaped(1, 1, 4)
        let routed = routing.route(
            logits: logits,
            expertBias: MLXArray.zeros([4]),
            outputDType: .float32
        )

        eval(routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [2])
    }

    @Test("constructs model with mixed caches, adapters, and greedy fast path")
    func constructsModelWithMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = AfMoEModel(Self.smallConfig(layerTypes: ["full_attention", "sliding_attention"]))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "afmoe")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? RotatingKVCache)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny AfMoE model produces finite logits")
    func tinyAfMoEModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = AfMoEModel(
                Self.smallConfig(layerTypes: ["full_attention", "sliding_attention"])
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs sparse per-expert checkpoint weights")
    func sanitizerPacksSparsePerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = AfMoEModel(
                Self.smallConfig(
                    layerTypes: ["full_attention", "sliding_attention"],
                    tieWordEmbeddings: true
                )
            )
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)

            let gate = try #require(sanitized["model.layers.1.mlp.experts.gate_proj.weight"])
            let down = try #require(sanitized["model.layers.1.mlp.experts.down_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.1.mlp.experts.up_proj.weight"]
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
        layerTypes: [String] = ["full_attention"],
        numExperts: Int = 2,
        numExpertsPerToken: Int = 1,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        routeScale: Float = 1,
        tieWordEmbeddings: Bool = false
    ) -> AfMoEConfiguration {
        AfMoEConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            hiddenLayers: layerTypes.count,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: 4,
            tieWordEmbeddings: tieWordEmbeddings,
            numExperts: numExperts,
            numExpertsPerToken: numExpertsPerToken,
            numSharedExperts: 1,
            numDenseLayers: 1,
            routeNorm: true,
            routeScale: routeScale,
            nGroup: nGroup,
            topkGroup: topkGroup,
            layerTypes: layerTypes,
            slidingWindow: 8,
            mupEnabled: false
        )
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
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
                weights["model.layers.1.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: baseValue + Float(expertIndex), count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }
}
