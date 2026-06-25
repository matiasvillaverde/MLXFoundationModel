import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Qwen3Next architecture")
struct Qwen3NextArchitectureTests {
    @Test("decodes default schedule from full-attention interval")
    func decodesDefaultScheduleFromInterval() throws {
        let json = #"""
        {
            "model_type": "qwen3_next",
            "num_hidden_layers": 4,
            "full_attention_interval": 2
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Qwen3NextConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "qwen3_next")
        #expect(config.layerTypes == [
            .linearAttention,
            .fullAttention,
            .linearAttention,
            .fullAttention
        ])
    }

    @Test("decodes explicit linear and full attention schedule")
    func decodesExplicitLayerSchedule() throws {
        let json = #"""
        {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention", "linear_attention"]
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Qwen3NextConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.layerTypes == [.fullAttention, .linearAttention])
        #expect(Qwen3NextLayerPlan(config).firstFullAttentionIndex == 0)
        #expect(Qwen3NextLayerPlan(config).firstLinearIndex == 1)
    }

    @Test("rejects layer schedule count mismatch")
    func rejectsLayerScheduleCountMismatch() {
        let json = #"""
        {
            "num_hidden_layers": 2,
            "layer_types": ["full_attention"]
        }
        """#

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                Qwen3NextConfiguration.self,
                from: Data(json.utf8)
            )
        }
    }

    @Test("computes full and linear attention layouts")
    func computesLayouts() {
        let config = Self.smallConfig(layerTypes: [.linearAttention, .fullAttention])
        let attention = Qwen3NextAttentionLayout(config)
        let linear = Qwen3NextLinearAttentionLayout(config)

        #expect(attention.queryProjectionDimensions == 16)
        #expect(attention.keyValueProjectionDimensions == 8)
        #expect(attention.rotaryDimensions == 2)
        #expect(linear.keyDimensions == 4)
        #expect(linear.valueDimensions == 8)
        #expect(linear.convolutionDimensions == 16)
        #expect(linear.valueHeadsPerKeyHead == 2)
    }

    @Test("cache types and LoRA targets follow layer schedule")
    func cacheTypesAndLoRATargetsFollowLayerSchedule() throws {
        let model = Qwen3NextModel(
            Self.smallConfig(layerTypes: [.linearAttention, .fullAttention])
        )
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(cache.count == 2)
        _ = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny Qwen3Next model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Qwen3NextModel(
                Self.smallConfig(layerTypes: [.linearAttention, .fullAttention])
            )
            let inputs = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(inputs, cache: [])

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs sparse experts beyond the first layer")
    func sanitizerPacksSparseExpertsBeyondFirstLayer() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Qwen3NextModel(
                Self.smallConfig(
                    layerTypes: [.linearAttention, .fullAttention],
                    numExperts: 2,
                    numExpertsPerTok: 1,
                    decoderSparseStep: 2,
                    tieWordEmbeddings: true
                )
            )
            let sanitized = model.sanitize(weights: Self.sparseCheckpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.mtp.fc.weight"] == nil)
            #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)

            let conv = try #require(sanitized["model.layers.0.linear_attn.conv1d.weight"])
            let norm = try #require(sanitized["model.layers.1.self_attn.q_norm.weight"])
            let gate = try #require(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
            let down = try #require(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]
            )

            eval(conv, norm, gate, down, upProjection)
            #expect(conv.shape == [2, 4, 3])
            #expect(norm.asArray(Float.self) == [2, 2, 2, 2])
            #expect(gate.shape == [2, 2, 2])
            #expect(down.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [6, 6, 6, 6])
        }
    }

    private static func smallConfig(
        layerTypes: [Qwen3NextLayerKind],
        numExperts: Int = 0,
        numExpertsPerTok: Int = 0,
        decoderSparseStep: Int = 1,
        tieWordEmbeddings: Bool = false
    ) -> Qwen3NextConfiguration {
        Qwen3NextConfiguration(
            hiddenSize: 16,
            hiddenLayers: layerTypes.count,
            intermediateSize: 32,
            attentionHeads: 4,
            linearNumValueHeads: 2,
            linearNumKeyHeads: 1,
            linearKeyHeadDim: 4,
            linearValueHeadDim: 4,
            linearConvKernelDim: 2,
            numExperts: numExperts,
            numExpertsPerTok: numExpertsPerTok,
            decoderSparseStep: decoderSparseStep,
            sharedExpertIntermediateSize: numExperts > 0 ? 8 : 0,
            moeIntermediateSize: numExperts > 0 ? 8 : 0,
            rmsNormEps: 1e-6,
            vocabularySize: 64,
            kvHeads: 2,
            ropeTheta: 10_000,
            partialRotaryFactor: 0.5,
            maxPositionEmbeddings: 128,
            normTopkProb: true,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionBias: false,
            headDim: 4,
            fullAttentionInterval: 2,
            layerTypes: layerTypes
        )
    }

    private static func sparseCheckpointWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.mtp.fc.weight": MLXArray.ones([2, 2]),
            "model.layers.0.linear_attn.conv1d.weight": MLXArray.ones([2, 3, 4]),
            "model.layers.1.self_attn.q_norm.weight": MLXArray.ones([4])
        ]
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
