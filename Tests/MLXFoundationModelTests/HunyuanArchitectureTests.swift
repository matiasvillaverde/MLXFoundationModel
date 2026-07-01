import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Hunyuan architecture")
struct HunyuanArchitectureTests {
    @Test("decodes Hunyuan 7B configuration")
    func decodesHunyuan7BConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            HunyuanConfiguration.self,
            from: Data(HunyuanTestFixtures.hunyuan7BConfigJSON.utf8)
        )
        let layout = HunyuanAttentionLayout(config)
        let layerPlan = HunyuanLayerPlan(config)

        #expect(config.modelType == "hunyuan")
        #expect(config.vocabularySize == 129_024)
        #expect(config.hiddenSize == 4_096)
        #expect(config.hiddenLayers == 32)
        #expect(config.intermediateSize == 14_336)
        #expect(config.attentionHeads == 32)
        #expect(config.kvHeads == 8)
        #expect(config.numExperts == 1)
        #expect(config.topK(layerIndex: 0) == 1)
        #expect(config.ropeAlpha == 1_000)
        #expect(config.tieWordEmbeddings)
        #expect(layout.headDim == 128)
        #expect(layout.keyValueProjectionSize == 1_024)
        #expect(!layerPlan.usesSparseExperts.contains(true))
    }

    @Test("decodes Hunyuan MoE layer metadata")
    func decodesHunyuanMoELayerMetadata() throws {
        let config = try JSONDecoder.json5().decode(
            HunyuanConfiguration.self,
            from: Data(HunyuanTestFixtures.hunyuanA13BMoEConfigJSON.utf8)
        )
        let layerPlan = HunyuanLayerPlan(config)
        let routing = HunyuanRoutingPlan(config, layerIndex: 0)

        #expect(config.numExperts == 64)
        #expect(config.topK(layerIndex: 31) == 8)
        #expect(config.sharedExpertCount(layerIndex: 0) == 1)
        #expect(config.expertIntermediateSize(layerIndex: 0) == 3_072)
        #expect(config.useMixedMLPMoE)
        #expect(routing.expertCount == 64)
        #expect(routing.topK == 8)
        #expect(!layerPlan.hasKeyValueProjection.contains(false))
        #expect(!layerPlan.usesSparseExperts.contains(false))
    }

    @Test("builds CLA projection plan")
    func buildsCLAProjectionPlan() {
        let config = Self.smallDenseConfig(hiddenLayers: 5, useCLA: true, claShareFactor: 2)
        let layerPlan = HunyuanLayerPlan(config)

        #expect(layerPlan.hasKeyValueProjection == [true, false, true, false, true])
    }

    @Test("router selects softmax top-k without renormalizing")
    func routerSelectsSoftmaxTopKWithoutRenormalizing() {
        let routing = HunyuanRoutingPlan(Self.smallMoEConfig(numExperts: 3, topK: 2), layerIndex: 0)
        let logits = MLXArray([Float(4), Float(2), Float(1)]).reshaped(1, 1, 3)
        let routed = routing.route(logits: logits, outputDType: .float32)
        let scoreSum = routed.scores.sum()

        eval(routed.indices, scoreSum)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [0, 1])
        #expect(scoreSum.item(Float.self) < 1)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = HunyuanModel(Self.smallDenseConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny dense model produces finite logits with cache")
    func tinyDenseModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = HunyuanModel(Self.smallDenseConfig())
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

    @Test("tiny MoE model produces finite logits")
    func tinyMoEModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = HunyuanModel(Self.smallMoEConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer strips tied head and packs legacy experts")
    func sanitizerStripsTiedHeadAndPacksLegacyExperts() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = HunyuanModel(Self.smallMoEConfig(hiddenLayers: 1, tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.legacyExpertWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)

            let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
            )
            eval(gate, upProjection)

            #expect(gate.shape == [3, 2, 2])
            #expect(upProjection.shape == [3, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [7, 7, 7, 7])
        }
    }

    private static func smallDenseConfig(
        hiddenLayers: Int = 1,
        useCLA: Bool = false,
        claShareFactor: Int = 2,
        tieWordEmbeddings: Bool = true
    ) -> HunyuanConfiguration {
        HunyuanConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            kvHeads: 2,
            useCLA: useCLA,
            claShareFactor: claShareFactor,
            ropeScaling: [
                "alpha": .float(1),
                "factor": .float(1),
                "type": .string("dynamic")
            ],
            tieWordEmbeddings: tieWordEmbeddings,
            headDim: 4
        )
    }

    private static func smallMoEConfig(
        hiddenLayers: Int = 1,
        numExperts: Int = 3,
        topK: Int = 2,
        tieWordEmbeddings: Bool = true
    ) -> HunyuanConfiguration {
        HunyuanConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            kvHeads: 2,
            moeTopK: HunyuanLayerInt(topK),
            numExperts: numExperts,
            numSharedExpert: HunyuanLayerInt(1),
            useMixedMLPMoE: true,
            moeIntermediateSize: HunyuanLayerInt(8),
            ropeScaling: [
                "alpha": .float(1),
                "factor": .float(1),
                "type": .string("dynamic")
            ],
            tieWordEmbeddings: tieWordEmbeddings,
            headDim: 4
        )
    }

    private static func legacyExpertWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2])
        ]
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(4)),
            ("up_proj", Float(7))
        ]

        for expertIndex in 0 ..< 3 {
            for (name, value) in projections {
                weights["model.layers.0.mlp.experts.\(expertIndex).\(name).weight"] = MLXArray(
                    [Float](repeating: value, count: 4)
                )
                .reshaped([2, 2])
            }
        }
        return weights
    }
}
