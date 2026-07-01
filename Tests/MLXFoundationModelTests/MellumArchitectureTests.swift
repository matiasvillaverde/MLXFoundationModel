import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Mellum architecture")
struct MellumArchitectureTests {
    @Test("decodes real-style layer and rope schedule")
    func decodesLayerAndRopeSchedule() throws {
        let config = try JSONDecoder.json5().decode(
            MellumConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )
        let plan = MellumLayerPlan(config)

        #expect(config.modelType == "mellum")
        #expect(config.hiddenLayers == 4)
        #expect(config.layerTypes == [.sliding, .sliding, .sliding, .full])
        #expect(plan.firstSlidingLayer == 0)
        #expect(plan.firstFullLayer == 3)
        #expect(config.slidingWindow == 1_024)
        #expect(config.ropeParameters["full_attention"]?["rope_type"] == .string("yarn"))
    }

    @Test("rejects mismatched layer schedule")
    func rejectsMismatchedLayerSchedule() {
        let json = #"""
        {
            "model_type": "mellum",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 8,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 128,
            "norm_topk_prob": true,
            "sliding_window": 16,
            "layer_types": ["full_attention"],
            "rope_parameters": {
                "full_attention": {"rope_type": "default", "rope_theta": 10000}
            }
        }
        """#

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(MellumConfiguration.self, from: Data(json.utf8))
        }
    }

    @Test("builds attention and router layouts")
    func buildsAttentionAndRouterLayouts() {
        let config = Self.smallConfig(layerTypes: [.sliding, .full])
        let attention = MellumAttentionLayout(config)
        let router = MellumRouterPlan(config)

        #expect(attention.queryHeads == 4)
        #expect(attention.keyValueHeads == 2)
        #expect(attention.headDimensions == 4)
        #expect(attention.queryProjectionSize == 16)
        #expect(attention.keyValueProjectionSize == 8)
        #expect(attention.attentionScale == 0.5)
        #expect(router.expertCount == 3)
        #expect(router.selectedExpertCount == 2)
        #expect(router.normalizesTopKProbabilities)
    }

    @Test("cache types and LoRA targets follow layer schedule")
    func cacheTypesAndLoRATargetsFollowLayerSchedule() throws {
        let model = MellumModel(Self.smallConfig(layerTypes: [.sliding, .full]))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "mellum")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny untied model produces finite logits with mixed cache")
    func tinyUntiedModelProducesFiniteLogitsWithMixedCache() {
        Device.withDefaultDevice(.cpu) {
            let model = MellumModel(Self.smallConfig(layerTypes: [.sliding, .full]))
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(cache[1].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny tied model produces finite logits")
    func tinyTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = MellumModel(
                Self.smallConfig(layerTypes: [.full], tieWordEmbeddings: true)
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs expert weights and drops tied output head")
    func sanitizerPacksExpertWeightsAndDropsTiedOutputHead() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = MellumModel(
                Self.smallConfig(layerTypes: [.full], tieWordEmbeddings: true)
            )
            let sanitized = model.sanitize(weights: Self.unpackedExpertWeights())

            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)

            let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
            let upProjection = try #require(
                sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]
            )
            let down = try #require(sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"])
            let scales = try #require(sanitized["model.layers.0.mlp.switch_mlp.up_proj.scales"])

            eval(gate, upProjection, down, scales)
            #expect(gate.shape == [3, 8, 16])
            #expect(upProjection.shape == [3, 8, 16])
            #expect(down.shape == [3, 16, 8])
            #expect(scales.shape == [3, 1])
        }
    }

    private static func smallConfig(
        layerTypes: [MellumLayerKind],
        tieWordEmbeddings: Bool = false
    ) -> MellumConfiguration {
        MellumConfiguration(
            hiddenSize: 16,
            hiddenLayers: layerTypes.count,
            intermediateSize: 32,
            attentionHeads: 4,
            numExperts: 3,
            numExpertsPerToken: 2,
            moeIntermediateSize: 8,
            vocabularySize: 64,
            kvHeads: 2,
            headDimensions: 4,
            tieWordEmbeddings: tieWordEmbeddings,
            maxPositionEmbeddings: 128,
            slidingWindow: 16,
            layerTypes: layerTypes,
            ropeParameters: [
                "full_attention": ["rope_type": .string("default"), "rope_theta": .int(10_000)],
                "sliding_attention": [
                    "rope_type": .string("default"),
                    "rope_theta": .int(10_000)
                ]
            ]
        )
    }

    private static var configJSON: String {
        #"""
        {
            "model_type": "mellum",
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "tie_word_embeddings": false,
            "max_position_embeddings": 128,
            "norm_topk_prob": true,
            "sliding_window": 1024,
            "layer_types": [
                "sliding_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention"
            ],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "yarn",
                    "rope_theta": 500000,
                    "factor": 16,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 32,
                    "beta_slow": 1
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 500000}
            }
        }
        """#
    }

    private static func unpackedExpertWeights() -> [String: MLXArray] {
        var weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([1])
        ]
        for expert in 0 ..< 3 {
            weights["model.layers.0.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.ones([8, 16]) * Float(expert + 1)
            weights["model.layers.0.mlp.experts.\(expert).up_proj.weight"] =
                MLXArray.ones([8, 16]) * Float(expert + 2)
            weights["model.layers.0.mlp.experts.\(expert).down_proj.weight"] =
                MLXArray.ones([16, 8]) * Float(expert + 3)
            weights["model.layers.0.mlp.experts.\(expert).up_proj.scales"] =
                MLXArray.ones([1]) * Float(expert + 4)
        }
        return weights
    }
}
