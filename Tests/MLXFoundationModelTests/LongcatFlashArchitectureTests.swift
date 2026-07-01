import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("LongCat Flash architecture")
struct LongcatFlashArchitectureTests {
    @Test("decodes LongCat Flash configuration")
    func decodesLongcatFlashConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            LongcatFlashConfiguration.self,
            from: Data(kLongcatFlashConfigJSON.utf8)
        )

        #expect(config.modelType == "longcat_flash")
        #expect(config.attentionMethod == "mla")
        #expect(config.zeroExpertType == "identity")
        #expect(config.hiddenSize == 6_144)
        #expect(config.feedForwardHiddenSize == 18_432)
        #expect(config.moeTopK == 8)
        #expect(config.routedExperts == 128)
        #expect(config.zeroExpertCount == 1)
        #expect(config.layerCount == 48)
        #expect(config.qLoraRank == 1_536)
        #expect(config.scaleQueryLora)
        #expect(config.scaleKeyValueLora)
    }

    @Test("registers LongCat Flash variants through the factory")
    func registersLongcatFlashVariantsThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("longcat_flash"))
        #expect(registeredTypes.contains("longcat_flash_ngram"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("LongcatFlashArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kLongcatFlashTinyConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "longcat_flash"
        )

        #expect(model is LongcatFlashModel)
        #expect((model as? LongcatFlashModel)?.vocabularySize == 64)
    }

    @Test("tiny LongCat Flash model produces finite logits")
    func tinyLongcatFlashModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = LongcatFlashModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect(model.newCache(parameters: nil).count == 2)
            #expect(model.newCache(parameters: nil).first is CacheList)
            #expect(model.loraLinearLayers().count == 4)
        }
    }

    @Test("tiny LongCat n-gram variant uses embedding context cache")
    func tinyLongcatNgramVariantUsesEmbeddingContextCache() {
        Device.withDefaultDevice(.cpu) {
            let model = LongcatFlashModel(Self.smallConfig(modelType: "longcat_flash_ngram"))
            let cache = model.newCache(parameters: nil)
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: cache)
            eval(logits)

            let token = model.greedyToken(
                LMInput.Text(tokens: MLXArray([4])),
                cache: cache,
                state: nil
            )
            eval(token.token)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect(cache.count == 3)
            #expect(cache.first is MambaCache)
            #expect((cache.first as? MambaCache)?[0]?.dim(-1) == 2)
            #expect(token.token.shape == [1])
        }
    }

    @Test("sanitizer packs experts and splits legacy MLA projections")
    func sanitizerPacksExpertsAndSplitsLegacyMLAProjections() {
        let model = LongcatFlashModel(Self.smallConfig())
        var weights: [String: MLXArray] = [
            "model.layers.0.self_attn.1.kv_b_proj.weight": MLXArray.ones([16, 6]),
            "model.layers.0.self_attn.1.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.mtp.blocks.0.weight": MLXArray.ones([1])
        ]

        for expert in 0 ..< 3 {
            weights["model.layers.0.mlp.experts.\(expert).gate_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.0.mlp.experts.\(expert).up_proj.weight"] = MLXArray.ones([8, 16])
            weights["model.layers.0.mlp.experts.\(expert).down_proj.weight"] = MLXArray.ones([16, 8])
        }

        let sanitized = model.sanitize(weights: weights)

        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape == [3, 8, 16])
        #expect(sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"]?.shape == [3, 8, 16])
        #expect(sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape == [3, 16, 8])
        #expect(sanitized["model.layers.0.self_attn.1.embed_q.weight"]?.shape == [2, 6, 4])
        #expect(sanitized["model.layers.0.self_attn.1.unembed_out.weight"]?.shape == [2, 4, 6])
        #expect(sanitized["model.layers.0.self_attn.1.kv_b_proj.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.1.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.mtp.blocks.0.weight"] == nil)
    }

    @Test("n-gram sanitizer moves token embeddings into the wrapper namespace")
    func ngramSanitizerMovesTokenEmbeddingsIntoWrapperNamespace() {
        let model = LongcatFlashModel(Self.smallConfig(modelType: "longcat_flash_ngram"))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.embed_tokens.weight"] == nil)
        #expect(sanitized["model.ngram_embeddings.word_embeddings.weight"]?.shape == [64, 16])
    }

    @Test("sanitizer preserves incomplete expert tensors")
    func sanitizerPreservesIncompleteExpertTensors() {
        let model = LongcatFlashModel(Self.smallConfig())
        let key = "model.layers.0.mlp.experts.0.gate_proj.weight"
        let sanitized = model.sanitize(weights: [
            key: MLXArray.ones([8, 16])
        ])

        #expect(sanitized[key]?.shape == [8, 16])
        #expect(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"] == nil)
    }

    private static func smallConfig(modelType: String = "longcat_flash") -> LongcatFlashConfiguration {
        LongcatFlashConfiguration(
            modelType: modelType,
            attentionMethod: "mla",
            hiddenSize: 16,
            feedForwardHiddenSize: 32,
            moeTopK: 2,
            expertFeedForwardHiddenSize: 8,
            routedExperts: 3,
            zeroExpertCount: 1,
            layerCount: 2,
            vocabularySize: 64,
            maxPositionEmbeddings: 64,
            attentionHeads: 2,
            kvLoraRank: 6,
            qLoraRank: 5,
            qkRopeHeadDim: 4,
            qkNopeHeadDim: 4,
            valueHeadDim: 4,
            routedScalingFactor: 1.5,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000,
            scaleQueryLora: true,
            scaleKeyValueLora: true,
            attentionBias: false,
            normalizeTopK: true,
            ngramVocabularySizeRatio: 2,
            embeddingNeighborCount: 3,
            embeddingSplitCount: 2
        )
    }
}

private let kLongcatFlashTinyConfigJSON = #"""
{
    "model_type": "longcat_flash",
    "attention_method": "mla",
    "zero_expert_type": "identity",
    "hidden_size": 16,
    "ffn_hidden_size": 32,
    "moe_topk": 2,
    "expert_ffn_hidden_size": 8,
    "n_routed_experts": 3,
    "zero_expert_num": 1,
    "num_layers": 2,
    "vocab_size": 64,
    "max_position_embeddings": 64,
    "num_attention_heads": 2,
    "kv_lora_rank": 6,
    "q_lora_rank": 5,
    "qk_rope_head_dim": 4,
    "qk_nope_head_dim": 4,
    "v_head_dim": 4,
    "routed_scaling_factor": 1.5,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000,
    "mla_scale_q_lora": true,
    "mla_scale_kv_lora": true,
    "attention_bias": false,
    "norm_topk_prob": true
}
"""#

private let kLongcatFlashConfigJSON = #"""
{
    "model_type": "longcat_flash",
    "attention_method": "mla",
    "zero_expert_type": "identity",
    "hidden_size": 6144,
    "ffn_hidden_size": 18432,
    "moe_topk": 8,
    "expert_ffn_hidden_size": 1536,
    "n_routed_experts": 128,
    "zero_expert_num": 1,
    "num_layers": 48,
    "vocab_size": 131072,
    "max_position_embeddings": 262144,
    "num_attention_heads": 48,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 128,
    "v_head_dim": 128,
    "routed_scaling_factor": 2.5,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "mla_scale_q_lora": true,
    "mla_scale_kv_lora": true,
    "attention_bias": false,
    "norm_topk_prob": true
}
"""#
