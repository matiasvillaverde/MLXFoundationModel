import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("EXAONE MoE architecture")
struct ExaoneMoEArchitectureTests {
    @Test("decodes EXAONE MoE config from current checkpoint metadata")
    func decodesConfigFromCurrentCheckpointMetadata() throws {
        let config = try JSONDecoder.json5().decode(
            ExaoneMoEConfiguration.self,
            from: Data(ExaoneMoETestFixtures.configJSON.utf8)
        )
        let layerPlan = ExaoneMoELayerPlan(config)

        #expect(config.modelType == "exaone_moe")
        #expect(config.vocabularySize == 153_600)
        #expect(config.hiddenSize == 2_048)
        #expect(config.intermediateSize == 6_144)
        #expect(config.moeIntermediateSize == 512)
        #expect(config.hiddenLayers == 16)
        #expect(config.attentionHeads == 16)
        #expect(config.kvHeads == 4)
        #expect(config.headDim == 128)
        #expect(config.ropeTheta == 1_000_000)
        #expect(config.ropeScaling?["rope_type"] == .string("default"))
        #expect(layerPlan.firstSlidingLayer == 0)
        #expect(layerPlan.firstFullLayer == 3)
        #expect(!layerPlan.usesSparseExperts(layerIndex: 0))
        #expect(layerPlan.usesSparseExperts(layerIndex: 1))
        #expect(layerPlan.attentionKind(layerIndex: 3) == .full)
    }

    @Test("builds attention, layer, and routing plans")
    func buildsAttentionLayerAndRoutingPlans() {
        let config = Self.smallConfig(
            layerTypes: ["sliding_attention", "full_attention"],
            usesMoELayer: [false, true],
            numExperts: 4,
            numExpertsPerToken: 2,
            nGroup: 2,
            topkGroup: 1
        )
        let layerPlan = ExaoneMoELayerPlan(config)
        let slidingAttention = ExaoneMoEAttentionLayout(
            config,
            kind: .sliding,
            applyRopeAllLayers: !layerPlan.hasSlidingAttention
        )
        let fullAttention = ExaoneMoEAttentionLayout(
            config,
            kind: .full,
            applyRopeAllLayers: !layerPlan.hasSlidingAttention
        )
        let routing = ExaoneMoERoutingPlan(config)

        expectSmallAttentionLayout(slidingAttention)
        #expect(slidingAttention.usesRotaryPosition)
        #expect(!fullAttention.usesRotaryPosition)
        #expect(!layerPlan.usesSparseExperts(layerIndex: 0))
        #expect(layerPlan.usesSparseExperts(layerIndex: 1))
        #expect(routing.expertsPerGroup == 2)
        #expect(routing.keptGroupCount == 1)
    }

    @Test("router uses correction bias for selection only")
    func routerUsesCorrectionBiasForSelectionOnly() {
        let routing = ExaoneMoERoutingPlan(
            Self.smallConfig(
                numExperts: 3,
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
        let routing = ExaoneMoERoutingPlan(
            Self.smallConfig(
                numExperts: 4,
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

    @Test("constructs model with mixed caches, adapters, and greedy fast path")
    func constructsModelWithMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = ExaoneMoEModel(
            Self.smallConfig(layerTypes: ["sliding_attention", "full_attention"])
        )
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? StandardKVCache)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny tied model produces finite logits with dense and sparse layers")
    func tinyTiedModelProducesFiniteLogitsWithDenseAndSparseLayers() {
        Device.withDefaultDevice(.cpu) {
            let model = ExaoneMoEModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = ExaoneMoEModel(Self.smallConfig(tieWordEmbeddings: false))
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer packs sparse experts and keeps dense layers independent")
    func sanitizerPacksSparseExpertsAndKeepsDenseLayersIndependent() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = ExaoneMoEModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(sanitized["mtp.proj.weight"] == nil)
            #expect(sanitized["model.layers.1.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
            #expect(sanitized["model.layers.1.mlp.e_score_correction_bias"] == nil)
            #expect(sanitized["model.layers.1.mlp.gate.e_score_correction_bias"] != nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] != nil)

            let gate = try #require(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
            let gateScales = try #require(
                sanitized["model.layers.1.mlp.switch_mlp.gate_proj.scales"]
            )
            let down = try #require(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
            )

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
        layerTypes: [String] = ["sliding_attention", "full_attention"],
        usesMoELayer: [Bool] = [false, true],
        numExperts: Int = 2,
        numExpertsPerToken: Int = 1,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        routedScalingFactor: Float = 1,
        tieWordEmbeddings: Bool = true
    ) -> ExaoneMoEConfiguration {
        ExaoneMoEConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            hiddenLayers: layerTypes.count,
            attentionHeads: 4,
            kvHeads: 2,
            headDim: 4,
            rmsNormEps: 1e-5,
            maxPositionEmbeddings: 64,
            slidingWindow: 4,
            layerTypes: layerTypes,
            usesMoELayer: usesMoELayer,
            numExperts: numExperts,
            numExpertsPerToken: numExpertsPerToken,
            numSharedExperts: 1,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: true,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private func expectSmallAttentionLayout(_ layout: ExaoneMoEAttentionLayout) {
        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDimensions == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "mtp.proj.weight": MLXArray.ones([2, 2]),
            "model.layers.1.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.1.mlp.e_score_correction_bias": MLXArray.zeros([2]),
            "model.layers.0.mlp.experts.0.gate_proj.weight": MLXArray.ones([2, 2])
        ]
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(3)),
            ("up_proj", Float(5))
        ]

        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in projections {
                insertExpertWeights(
                    projection: name,
                    value: baseValue + Float(expertIndex),
                    expertIndex: expertIndex,
                    into: &weights
                )
            }
        }

        return weights
    }

    private static func insertExpertWeights(
        projection: String,
        value: Float,
        expertIndex: Int,
        into weights: inout [String: MLXArray]
    ) {
        let prefix = "model.layers.1.mlp.experts.\(expertIndex).\(projection)"
        weights["\(prefix).weight"] = MLXArray([value, value, value, value]).reshaped(2, 2)
        weights["\(prefix).scales"] = MLXArray([value])
        weights["\(prefix).biases"] = MLXArray([value])
    }
}
