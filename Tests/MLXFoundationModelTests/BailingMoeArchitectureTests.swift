import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Bailing MoE architecture")
struct BailingMoeArchitectureTests {
    @Test("decodes Bailing MoE configuration with project defaults")
    func decodesBailingMoeConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "num_experts": 2,
            "num_attention_heads": 4,
            "num_experts_per_tok": 1,
            "num_hidden_layers": 1,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            BailingMoeConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "bailing_moe")
        #expect(config.kvHeads == 4)
        #expect(config.numSharedExperts == 0)
        #expect(config.firstKDenseReplace == 0)
        #expect(config.ropeTheta == 10_000)
        #expect(config.scoreFunction == "softmax")
        #expect(config.nGroup == 1)
        #expect(config.topkGroup == 1)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("builds attention, layer, and routing plans")
    func buildsAttentionLayerAndRoutingPlans() {
        let config = Self.smallConfig(
            hiddenLayers: 3,
            kvHeads: 2,
            partialRotaryFactor: 0.5,
            firstKDenseReplace: 1,
            nGroup: 2,
            topkGroup: 1
        )
        let attention = BailingMoeAttentionLayout(config)
        let layers = BailingMoeLayerPlan(config)
        let routing = BailingMoeRoutingPlan(config)

        #expect(attention.hiddenSize == 16)
        #expect(attention.attentionHeads == 4)
        #expect(attention.keyValueHeads == 2)
        #expect(attention.headDimensions == 4)
        #expect(attention.rotaryDimensions == 2)
        #expect(attention.queryKeyValueProjectionSize == 32)
        #expect(attention.attentionScale == 0.5)
        #expect(layers.usesSparseExperts(layerIndex: 0) == false)
        #expect(layers.usesSparseExperts(layerIndex: 1) == true)
        #expect(routing.expertsPerGroup == 1)
        #expect(routing.keptGroupCount == 1)
    }

    @Test("router uses expert bias for selection only")
    func routerUsesExpertBiasForSelectionOnly() {
        let routing = BailingMoeRoutingPlan(
            Self.smallConfig(
                numExperts: 3,
                numExpertsPerToken: 1,
                moeRouterEnableExpertBias: true,
                scoreFunction: "sigmoid",
                routedScalingFactor: 2
            )
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
        let routing = BailingMoeRoutingPlan(
            Self.smallConfig(
                numExperts: 4,
                numExpertsPerToken: 1,
                nGroup: 2,
                topkGroup: 1,
                scoreFunction: "sigmoid"
            )
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

    @Test("constructs Bailing MoE model with cache, adapters, and greedy fast path")
    func constructsBailingMoeModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = BailingMoeModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "bailing_moe")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["query_key_value"])
    }

    @Test("tiny Bailing MoE model produces finite logits")
    func tinyBailingMoeModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = BailingMoeModel(Self.smallConfig(tieWordEmbeddings: false))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs per-expert checkpoint weights")
    func sanitizerPacksPerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = BailingMoeModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)

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
        kvHeads: Int? = 2,
        partialRotaryFactor: Float = 1,
        firstKDenseReplace: Int = 0,
        numExperts: Int = 2,
        numExpertsPerToken: Int = 1,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        moeRouterEnableExpertBias: Bool = false,
        scoreFunction: String = "softmax",
        routedScalingFactor: Float = 1,
        tieWordEmbeddings: Bool = false
    ) -> BailingMoeConfiguration {
        BailingMoeConfiguration(
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            numExperts: numExperts,
            numSharedExperts: 1,
            normTopkProb: true,
            attentionHeads: 4,
            numExpertsPerToken: numExpertsPerToken,
            hiddenLayers: hiddenLayers,
            kvHeads: kvHeads,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            firstKDenseReplace: firstKDenseReplace,
            useQKNorm: true,
            tieWordEmbeddings: tieWordEmbeddings,
            partialRotaryFactor: partialRotaryFactor,
            moeRouterEnableExpertBias: moeRouterEnableExpertBias,
            routedScalingFactor: routedScalingFactor,
            scoreFunction: scoreFunction,
            nGroup: nGroup,
            topkGroup: topkGroup
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
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: baseValue + Float(expertIndex), count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }
}
