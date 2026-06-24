import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("DeepSeek V3 architecture")
struct DeepseekV3ArchitectureTests {
    @Test("decodes DeepSeek V3 configuration with project defaults")
    func decodesDeepseekV3ConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "kv_lora_rank": 6,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "q_lora_rank": 0,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            DeepseekV3Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.vocabSize == 64)
        #expect(config.hiddenSize == 16)
        #expect(config.intermediateSize == 32)
        #expect(config.moeIntermediateSize == 32)
        #expect(config.numAttentionHeads == 2)
        #expect(config.numKeyValueHeads == 2)
        #expect(config.qLoraRank == nil)
        #expect(config.maxPositionEmbeddings == 4_096)
        #expect(config.ropeTheta == 10_000)
        #expect(config.attentionBias == false)
    }

    @Test("builds DeepSeek V3 attention layout")
    func buildsDeepseekV3AttentionLayout() {
        let layout = DeepseekV3AttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 2)
        #expect(layout.kvLoraRank == 6)
        #expect(layout.queryLowRank == 5)
        #expect(layout.nopeHeadSize == 4)
        #expect(layout.ropeHeadSize == 4)
        #expect(layout.valueHeadSize == 4)
        #expect(layout.queryHeadSize == 8)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.compressedKeyValueSize == 10)
        #expect(layout.keyValueProjectionSize == 16)
        #expect(layout.outputProjectionSize == 8)
        #expect(layout.attentionScale == 0.35355338)
    }

    @Test("plans DeepSeek YaRN rotary scaling")
    func plansDeepseekYarnRotaryScaling() {
        let config = Self.smallConfig(
            ropeScaling: [
                "factor": .float(4),
                "original_max_position_embeddings": .int(64),
                "beta_fast": .float(16),
                "beta_slow": .float(2),
                "mscale": .float(1),
                "mscale_all_dim": .float(1)
            ]
        )

        let plan = DeepseekV3YarnPlan(config, dimensions: 4)
        let range = plan.correctionRange()

        #expect(plan.scalingFactor == 4)
        #expect(plan.originalMaxPositionEmbeddings == 64)
        #expect(plan.rotaryInputScale == 1)
        #expect(plan.attentionScaleMultiplier > 1)
        #expect(range.low >= 0)
        #expect(range.high <= 3)
    }

    @Test("plans grouped expert routing")
    func plansGroupedExpertRouting() {
        let plan = DeepseekV3RoutingPlan(Self.smallConfig())

        #expect(plan.routedExperts == 4)
        #expect(plan.expertsPerToken == 2)
        #expect(plan.groupCount == 2)
        #expect(plan.keptGroupCount == 1)
        #expect(plan.expertsPerGroup == 2)
        #expect(plan.droppedGroupCount == 1)
        #expect(plan.normalizeTopK)
        #expect(plan.routedScalingFactor == 1.5)
    }

    @Test("constructs DeepSeek V3 model with cache and adapter targets")
    func constructsDeepseekV3ModelWithCacheAndAdapterTargets() {
        let model = DeepseekV3Model(Self.smallConfig())
        let noQueryLoraModel = DeepseekV3Model(Self.smallConfig(qLoraRank: nil))
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(model.loraLinearLayers().count == 2)
        #expect(model.loraLinearLayers()[0].1 == [
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj"
        ])
        #expect(noQueryLoraModel.loraLinearLayers()[0].1 == [
            "q_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj"
        ])
    }

    @Test("tiny DeepSeek V3 model produces finite logits")
    func tinyDeepseekV3ModelProducesFiniteLogits() {
        let model = DeepseekV3Model(Self.smallConfig(nRoutedExperts: nil))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer packs expert weights and drops unused checkpoint tensors")
    func sanitizerPacksExpertWeightsAndDropsUnusedCheckpointTensors() {
        let model = DeepseekV3Model(Self.smallConfig())
        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.2.self_attn.q_proj.weight": MLXArray.ones([16, 16])
        ]

        for expert in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.1.mlp.experts.\(expert).up_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.1.mlp.experts.\(expert).down_proj.weight"] = MLXArray.ones([16, 8])
        }

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 8, 16])
        #expect(sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape == [4, 8, 16])
        #expect(sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape == [4, 16, 8])
        #expect(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.2.self_attn.q_proj.weight"] == nil)
    }

    private static func smallConfig(
        qLoraRank: Int? = 5,
        nRoutedExperts: Int? = 4,
        ropeScaling: [String: StringOrNumber] = [:]
    ) -> DeepseekV3Configuration {
        DeepseekV3Configuration(
            vocabSize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            nSharedExperts: nRoutedExperts == nil ? nil : 1,
            nRoutedExperts: nRoutedExperts,
            routedScalingFactor: 1.5,
            kvLoraRank: 6,
            qLoraRank: qLoraRank,
            qkRopeHeadDim: 4,
            vHeadDim: 4,
            qkNopeHeadDim: 4,
            normTopkProb: true,
            nGroup: nRoutedExperts == nil ? nil : 2,
            topkGroup: nRoutedExperts == nil ? nil : 1,
            numExpertsPerTok: nRoutedExperts == nil ? nil : 2,
            moeLayerFreq: 1,
            firstKDenseReplace: 1,
            maxPositionEmbeddings: 64,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            attentionBias: false
        )
    }
}
