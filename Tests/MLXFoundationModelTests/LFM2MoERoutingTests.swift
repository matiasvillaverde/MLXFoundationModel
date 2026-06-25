import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("LFM2 MoE routing")
struct LFM2MoERoutingTests {
    @Test("decodes layer plan from layer types")
    func decodesLayerPlanFromLayerTypes() throws {
        let config = try Self.decodeConfig()

        #expect(config.layerPlan.kinds == [
            .convolution,
            .fullAttention,
            .convolution,
            .fullAttention
        ])
        #expect(config.fullAttnIdxs == [1, 3])
        #expect(config.layerPlan.firstConvolutionIndex == 0)
        #expect(config.layerPlan.firstAttentionIndex == 1)
    }

    @Test("explicit full attention indexes override layer types")
    func explicitFullAttentionIndexesOverrideLayerTypes() throws {
        let config = try Self.decodeConfig(fullAttentionIndices: [0, 2])

        #expect(config.fullAttnIdxs == [0, 2])
        #expect(config.layerPlan.kinds == [
            .fullAttention,
            .convolution,
            .fullAttention,
            .convolution
        ])
    }

    @Test("computes attention and convolution layouts")
    func computesLayouts() throws {
        let config = try Self.decodeConfig()
        let attention = LFM2MoEAttentionLayout(config)
        let convolution = LFM2MoEConvolutionLayout(config)

        #expect(attention.headDimensions == 4)
        #expect(attention.queryDimensions == 16)
        #expect(attention.keyValueDimensions == 8)
        #expect(convolution.stateLength == 2)
        #expect(convolution.projectionDimensions == 48)
    }

    @Test("cache types follow layer plan")
    func cacheTypesFollowLayerPlan() throws {
        let model = LFM2MoEModel(try Self.decodeConfig())
        let cache = model.newCache(parameters: nil)

        #expect(cache.count == 4)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is KVCacheSimple)
        #expect(cache[2] is MambaCache)
        #expect(cache[3] is KVCacheSimple)
    }

    @Test("LoRA targets only attention projections")
    func loraTargetsOnlyAttentionProjections() throws {
        let model = LFM2MoEModel(try Self.decodeConfig())
        let targets = model.loraLinearLayers()

        #expect(targets.count == 2)
        #expect(targets.allSatisfy { $0.1 == ["q_proj", "v_proj"] })
    }

    @Test("uses sigmoid scores and keeps expert bias selection-only")
    func usesSigmoidScoresAndKeepsExpertBiasSelectionOnly() {
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)
        let expertBias = MLXArray([Float(0), Float(10), Float(0)])

        let routed = lfm2MoERouter(
            logits: logits,
            expertBias: expertBias,
            topK: 1,
            normTopKProb: false,
            useExpertBias: true,
            routedScalingFactor: 2
        )

        eval(routed.indices, routed.scores)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [1])
        #expect(abs(routed.scores.item(Float.self) - (sigmoid(1) * 2)) < 0.0001)
    }

    @Test("normalizes selected sigmoid scores before routed scaling")
    func normalizesSelectedSigmoidScoresBeforeRoutedScaling() {
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)

        let routed = lfm2MoERouter(
            logits: logits,
            expertBias: nil,
            topK: 2,
            normTopKProb: true,
            useExpertBias: false,
            routedScalingFactor: 1.5
        )

        eval(routed.scores)
        let scores = routed.scores.asArray(Float.self)

        #expect(abs(scores.reduce(0, +) - 1.5) < 0.0001)
    }

    @Test("sanitizer renames and packs expert projections")
    func sanitizerRenamesAndPacksExpertProjections() throws {
        let model = LFM2MoEModel(try Self.decodeConfig())
        let prefix = "model.layers.2.feed_forward.experts"
        let sanitized = model.sanitize(weights: [
            "\(prefix).0.w1.weight": MLXArray.zeros([3, 4]),
            "\(prefix).1.w1.weight": MLXArray.ones([3, 4]),
            "\(prefix).0.w2.weight": MLXArray.zeros([4, 3]),
            "\(prefix).1.w2.weight": MLXArray.ones([4, 3]),
            "\(prefix).0.w3.weight": MLXArray.zeros([3, 4]),
            "\(prefix).1.w3.weight": MLXArray.ones([3, 4]),
            "model.layers.0.conv.conv.weight": MLXArray.zeros([1, 2, 4])
        ])

        #expect(
            sanitized["model.layers.2.feed_forward.switch_mlp.gate_proj.weight"]?.shape
                == [2, 3, 4]
        )
        #expect(
            sanitized["model.layers.2.feed_forward.switch_mlp.down_proj.weight"]?.shape
                == [2, 4, 3]
        )
        #expect(sanitized["model.layers.0.conv.conv.weight"]?.shape == [1, 4, 2])
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }

    private static func decodeConfig(
        fullAttentionIndices: [Int] = []
    ) throws -> LFM2MoEConfiguration {
        try JSONDecoder.json5().decode(
            LFM2MoEConfiguration.self,
            from: configJSON(fullAttentionIndices: fullAttentionIndices)
        )
    }

    private static func configJSON(fullAttentionIndices: [Int]) -> Data {
        let fullAttentionField = Self.fullAttentionJSONField(fullAttentionIndices)
        return Data(
            """
            {
                "conv_L_cache": 3,
                "conv_bias": false,
                \(fullAttentionField)
                "hidden_size": 16,
                "intermediate_size": 32,
                "layer_types": ["conv", "full_attention", "conv", "full_attention"],
                "max_position_embeddings": 64,
                "model_type": "lfm2_moe",
                "moe_intermediate_size": 8,
                "norm_eps": 0.00001,
                "norm_topk_prob": true,
                "num_attention_heads": 4,
                "num_dense_layers": 1,
                "num_experts": 2,
                "num_experts_per_tok": 1,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "routed_scaling_factor": 1.0,
                "rope_parameters": {"rope_theta": 1000000},
                "use_expert_bias": true,
                "vocab_size": 64
            }
            """.utf8
        )
    }

    private static func fullAttentionJSONField(_ indices: [Int]) -> String {
        guard !indices.isEmpty else {
            return ""
        }
        return "\"full_attn_idxs\": [\(indices.map(String.init).joined(separator: ", "))],"
    }
}
