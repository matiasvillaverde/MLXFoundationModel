import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GLM4 MoE architecture")
struct GLM4MoEArchitectureTests {
    @Test("decodes GLM4 MoE configuration with project defaults")
    func decodesGLM4MoEConfigurationWithDefaults() throws {
        let json = #"""
        {
            "model_type": "glm4_moe",
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 1
        }
        """#

        let config = try JSONDecoder.json5().decode(
            GLM4MoEConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "glm4_moe")
        #expect(config.maxPositionEmbeddings == 32_768)
        #expect(config.moeIntermediateSize == 32)
        #expect(config.kvHeads == 4)
        #expect(config.headDim == 4)
        #expect(config.nGroup == 1)
        #expect(config.topkGroup == 1)
        #expect(config.scoringFunc == "sigmoid")
        #expect(config.topkMethod == "noaux_tc")
    }

    @Test("builds attention, layer, and routing plans")
    func buildsAttentionLayerAndRoutingPlans() {
        let config = Self.smallConfig(
            hiddenLayers: 3,
            nRoutedExperts: 4,
            firstKDenseReplace: 1,
            kvHeads: 2,
            partialRotaryFactor: 0.5,
            nGroup: 2,
            topkGroup: 1
        )
        let attention = GLM4MoEAttentionLayout(config)
        let layers = GLM4MoELayerPlan(config)
        let routing = GLM4MoERoutingPlan(config)

        #expect(attention.hiddenSize == 16)
        #expect(attention.attentionHeads == 4)
        #expect(attention.keyValueHeads == 2)
        #expect(attention.headDimensions == 4)
        #expect(attention.rotaryDimensions == 2)
        #expect(attention.queryProjectionSize == 16)
        #expect(attention.keyValueProjectionSize == 8)
        #expect(attention.attentionScale == 0.5)
        #expect(layers.usesSparseExperts(layerIndex: 0) == false)
        #expect(layers.usesSparseExperts(layerIndex: 1) == true)
        #expect(routing.expertsPerGroup == 2)
        #expect(routing.keptGroupCount == 1)
    }

    @Test("router uses correction bias for selection only")
    func routerUsesCorrectionBiasForSelectionOnly() {
        let routing = GLM4MoERoutingPlan(
            Self.smallConfig(
                nRoutedExperts: 3,
                routedScalingFactor: 2
            )
        )
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)
        let bias = MLXArray([Float(0), Float(10), Float(0)])
        let routed = routing.route(logits: logits, correctionBias: bias, outputDType: .float32)

        eval(routed.scores, routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [1])
        #expect(abs(routed.scores.item(Float.self) - (sigmoid(1) * 2)) < 0.0001)
    }

    @Test("router masks lower-scoring groups before expert selection")
    func routerMasksLowerScoringGroupsBeforeExpertSelection() {
        let routing = GLM4MoERoutingPlan(
            Self.smallConfig(
                nRoutedExperts: 4,
                nGroup: 2,
                topkGroup: 1
            )
        )
        let logits = MLXArray([Float(1), Float(4), Float(3), Float(2)]).reshaped(1, 1, 4)
        let routed = routing.route(
            logits: logits,
            correctionBias: MLXArray.zeros([4]),
            outputDType: .float32
        )

        eval(routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [2])
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = GLM4MoEModel(Self.smallConfig(hiddenLayers: 2, nRoutedExperts: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "glm4_moe")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny GLM4 MoE model produces finite logits")
    func tinyGLM4MoEModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = GLM4MoEModel(Self.smallConfig(nRoutedExperts: 2))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs per-expert checkpoint weights")
    func sanitizerPacksPerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GLM4MoEModel(
                Self.smallConfig(nRoutedExperts: 2, tieWordEmbeddings: true)
            )
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
            #expect(sanitized["model.layers.1.extra.weight"] == nil)

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
        nRoutedExperts: Int? = 2,
        firstKDenseReplace: Int = 0,
        kvHeads: Int? = 2,
        partialRotaryFactor: Float = 1,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        routedScalingFactor: Float = 1,
        tieWordEmbeddings: Bool = false
    ) -> GLM4MoEConfiguration {
        GLM4MoEConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            normTopkProb: true,
            attentionHeads: 4,
            nGroup: nGroup,
            topkGroup: topkGroup,
            nSharedExperts: 1,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: routedScalingFactor,
            numExpertsPerTok: 1,
            firstKDenseReplace: firstKDenseReplace,
            hiddenLayers: hiddenLayers,
            kvHeads: kvHeads,
            useQkNorm: true,
            tieWordEmbeddings: tieWordEmbeddings,
            partialRotaryFactor: partialRotaryFactor
        )
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.1.extra.weight": MLXArray.ones([1])
        ]
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
