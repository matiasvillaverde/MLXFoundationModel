import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Klear architecture")
struct KlearArchitectureTests {
    @Test("decodes Klear config from current checkpoint metadata")
    func decodesConfigFromCurrentCheckpointMetadata() throws {
        let config = try JSONDecoder.json5().decode(
            KlearConfiguration.self,
            from: Data(KlearTestFixtures.configJSON.utf8)
        )
        let attention = KlearAttentionLayout(config)
        let layerPlan = KlearLayerPlan(config)

        #expect(config.modelType == "Klear")
        #expect(config.vocabularySize == 151_936)
        #expect(config.hiddenSize == 2_048)
        #expect(config.hiddenLayers == 32)
        #expect(config.intermediateSize == 8_064)
        #expect(config.moeIntermediateSize == 896)
        #expect(config.attentionHeads == 32)
        #expect(config.kvHeads == 4)
        #expect(config.numExperts == 256)
        #expect(config.numExpertsPerToken == 8)
        #expect(config.numSharedExperts == 1)
        #expect(config.ropeTheta == 500_000)
        #expect(!config.tieWordEmbeddings)
        #expect(attention.headSize == 64)
        #expect(attention.keyValueDimensions == 256)
        #expect(!layerPlan.usesSparseExperts.contains(false))
    }

    @Test("builds dense and sparse layer plans")
    func buildsDenseAndSparseLayerPlans() {
        let config = Self.smallConfig(
            hiddenLayers: 4,
            mlpOnlyLayers: [0],
            decoderSparseStep: 2
        )
        let layerPlan = KlearLayerPlan(config)

        #expect(layerPlan.usesSparseExperts == [false, true, false, true])
    }

    @Test("router uses expert bias for selection only")
    func routerUsesExpertBiasForSelectionOnly() {
        let routing = KlearRoutingPlan(
            Self.smallConfig(numExperts: 3, numExpertsPerToken: 1, normTopkProb: false)
        )
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)
        let expertBias = MLXArray([Float(0), Float(10), Float(0)])
        let routed = routing.route(logits: logits, expertBias: expertBias, outputDType: .float32)

        eval(routed.indices, routed.scores)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [1])
        #expect(abs(routed.scores.item(Float.self) - sigmoid(1)) < 0.0001)
    }

    @Test("router normalizes selected sigmoid scores")
    func routerNormalizesSelectedSigmoidScores() {
        let routing = KlearRoutingPlan(
            Self.smallConfig(numExperts: 3, numExpertsPerToken: 2, normTopkProb: true)
        )
        let logits = MLXArray([Float(4), Float(2), Float(1)]).reshaped(1, 1, 3)
        let routed = routing.route(
            logits: logits,
            expertBias: MLXArray.zeros([3]),
            outputDType: .float32
        )
        let scoreSum = routed.scores.sum()

        eval(scoreSum)

        #expect(abs(scoreSum.item(Float.self) - 1) < 0.0001)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = KlearModel(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny tied model produces finite logits with dense and sparse layers")
    func tinyTiedModelProducesFiniteLogitsWithDenseAndSparseLayers() {
        Device.withDefaultDevice(.cpu) {
            let model = KlearModel(Self.smallConfig(mlpOnlyLayers: [0], tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = KlearModel(Self.smallConfig(tieWordEmbeddings: false))
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

    @Test("sanitizer keeps quantized router and expert tensors")
    func sanitizerKeepsQuantizedRouterAndExpertTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = KlearModel(Self.smallConfig(tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(sanitized["model.layers.0.mlp.gate.scales"] != nil)
            #expect(sanitized["model.layers.0.mlp.gate.biases"] != nil)
            #expect(sanitized["model.layers.0.mlp.experts.gate_proj.scales"] != nil)

            let gate = try #require(sanitized["model.layers.0.mlp.gate.weight"])
            let expertGate = try #require(
                sanitized["model.layers.0.mlp.experts.gate_proj.weight"]
            )

            eval(gate, expertGate)
            #expect(gate.shape == [4, 16])
            #expect(expertGate.shape == [4, 8, 16])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 2,
        mlpOnlyLayers: [Int] = [],
        numExperts: Int = 4,
        numExpertsPerToken: Int = 2,
        decoderSparseStep: Int = 1,
        normTopkProb: Bool = true,
        tieWordEmbeddings: Bool = false
    ) -> KlearConfiguration {
        KlearConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            attentionHeads: 4,
            kvHeads: 2,
            rmsNormEps: 1e-5,
            mlpOnlyLayers: mlpOnlyLayers,
            numExperts: numExperts,
            numExpertsPerToken: numExpertsPerToken,
            decoderSparseStep: decoderSparseStep,
            numSharedExperts: 1,
            normTopkProb: normTopkProb,
            ropeTheta: 10_000,
            maxPositionEmbeddings: 128,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        let gateWeight = MLXArray(0 ..< 256).asType(.float32).reshaped(4, 64)
        let (quantizedGate, gateScales, gateBiases) = MLX.quantized(
            gateWeight,
            groupSize: 64,
            bits: 8,
            mode: .affine
        )

        var weights = [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([64, 1]),
            "lm_head.biases": MLXArray.zeros([64, 1]),
            "model.layers.0.mlp.gate.weight": quantizedGate,
            "model.layers.0.mlp.gate.scales": gateScales,
            "model.layers.0.mlp.experts.gate_proj.weight": MLXArray.ones([4, 8, 16]),
            "model.layers.0.mlp.experts.gate_proj.scales": MLXArray.ones([4, 8, 1])
        ]
        weights["model.layers.0.mlp.gate.biases"] = gateBiases
        return weights
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }
}
