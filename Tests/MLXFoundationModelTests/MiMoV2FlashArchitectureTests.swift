import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MiMo v2 Flash architecture")
struct MiMoV2FlashArchitectureTests {
    @Test("decodes MiMo v2 Flash configuration with project defaults")
    func decodesMiMoV2FlashConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "hybrid_layer_pattern": [0, 1],
            "moe_layer_freq": [0, 0]
        }
        """#

        let config = try JSONDecoder.json5().decode(
            MiMoV2FlashConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "mimo_v2_flash")
        #expect(config.numExpertsPerTok == 1)
        #expect(config.slidingWindowSize == 4_096)
        #expect(config.headDim == 4)
        #expect(config.vHeadDim == 4)
        #expect(config.swaAttentionHeads == 4)
        #expect(config.swaKvHeads == 2)
        #expect(config.swaRopeTheta == 10_000)
        #expect(config.topkGroup == 1)
    }

    @Test("builds full and sliding attention layouts")
    func buildsFullAndSlidingAttentionLayouts() {
        let config = Self.smallConfig(
            hybridLayerPattern: [0, 1],
            moeLayerFreq: [0, 1],
            nRoutedExperts: 4,
            nGroup: 2,
            topkGroup: 1,
            partialRotaryFactor: 0.5,
            addFullAttentionSinkBias: true,
            swaAttentionHeads: 2,
            swaKvHeads: 1,
            swaHeadDim: 8,
            swaVHeadDim: 8
        )
        let full = MiMoV2FlashAttentionLayout(config, kind: .full)
        let sliding = MiMoV2FlashAttentionLayout(config, kind: .sliding)

        #expect(full.attentionHeads == 4)
        #expect(full.keyValueHeads == 2)
        #expect(full.headDimensions == 4)
        #expect(full.rotaryDimensions == 2)
        #expect(full.usesAttentionSink == true)
        #expect(sliding.attentionHeads == 2)
        #expect(sliding.keyValueHeads == 1)
        #expect(sliding.headDimensions == 8)
        #expect(sliding.valueHeadDimensions == 8)
    }

    @Test("builds layer schedule and routing plan")
    func buildsLayerScheduleAndRoutingPlan() {
        let config = Self.smallConfig(
            hybridLayerPattern: [0, 1],
            moeLayerFreq: [0, 1],
            nRoutedExperts: 4,
            nGroup: 2,
            topkGroup: 1
        )
        let schedule = MiMoV2FlashLayerSchedule(config)
        let routing = MiMoV2FlashRoutingPlan(config)

        #expect(schedule.attentionKind(layerIndex: 0) == .full)
        #expect(schedule.attentionKind(layerIndex: 1) == .sliding)
        #expect(schedule.usesMoE(layerIndex: 1) == true)
        #expect(routing.expertsPerGroup == 2)
        #expect(routing.keptGroupCount == 1)
    }

    @Test("router masks lower-scoring groups before expert selection")
    func routerMasksLowerScoringGroupsBeforeExpertSelection() {
        let routing = MiMoV2FlashRoutingPlan(
            Self.smallConfig(
                nRoutedExperts: 4,
                nGroup: 2,
                topkGroup: 1,
                routedScalingFactor: 2
            )
        )
        let logits = MLXArray([Float(1), Float(4), Float(3), Float(2)]).reshaped(1, 1, 4)
        let routed = routing.route(
            logits: logits,
            correctionBias: MLXArray.zeros([4]),
            outputDType: .float32
        )

        eval(routed.scores, routed.indices)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [2])
        #expect(abs(routed.scores.item(Float.self) - (sigmoid(3) * 2)) < 0.0001)
    }

    @Test("constructs model with mixed caches, adapters, and greedy fast path")
    func constructsModelWithMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = MiMoV2FlashModel(
            Self.smallConfig(
                hybridLayerPattern: [0, 1],
                moeLayerFreq: [0, 0],
                swaKvHeads: 1
            )
        )
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "mimo_v2_flash")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 1])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? RotatingKVCache)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny MiMo v2 Flash model produces finite logits")
    func tinyMiMoV2FlashModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = MiMoV2FlashModel(
                Self.smallConfig(hybridLayerPattern: [0], moeLayerFreq: [0])
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs per-expert checkpoint weights")
    func sanitizerPacksPerExpertCheckpointWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = MiMoV2FlashModel(
                Self.smallConfig(
                    hybridLayerPattern: [0],
                    moeLayerFreq: [1],
                    nRoutedExperts: 2
                )
            )
            let sanitized = model.sanitize(weights: Self.expertCheckpointWeights())

            #expect(sanitized["model.mtp.foo"] == nil)
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
        hybridLayerPattern: [Int] = [0],
        moeLayerFreq: [Int] = [0],
        nRoutedExperts: Int? = nil,
        nGroup: Int = 1,
        topkGroup: Int? = nil,
        routedScalingFactor: Float? = nil,
        partialRotaryFactor: Float = 1,
        addFullAttentionSinkBias: Bool = false,
        swaAttentionHeads: Int? = nil,
        swaKvHeads: Int? = nil,
        swaHeadDim: Int? = nil,
        swaVHeadDim: Int? = nil
    ) -> MiMoV2FlashConfiguration {
        MiMoV2FlashConfiguration(
            hybridLayerPattern: hybridLayerPattern,
            moeLayerFreq: moeLayerFreq,
            addFullAttentionSinkBias: addFullAttentionSinkBias,
            slidingWindowSize: 8,
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            hiddenLayers: hybridLayerPattern.count,
            attentionHeads: 4,
            kvHeads: 2,
            nSharedExperts: 1,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: true,
            nGroup: nGroup,
            topkGroup: topkGroup,
            swaAttentionHeads: swaAttentionHeads,
            swaKvHeads: swaKvHeads,
            swaHeadDim: swaHeadDim,
            swaVHeadDim: swaVHeadDim,
            partialRotaryFactor: partialRotaryFactor
        )
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func expertCheckpointWeights() -> [String: MLXArray] {
        var weights = ["model.mtp.foo": MLXArray.ones([1])]
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
