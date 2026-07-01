import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Kimi K2.5 architecture")
struct KimiK25ArchitectureTests {
    @Test("decodes Kimi K2.5 wrapper configuration")
    func decodesKimiK25WrapperConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            KimiK25Configuration.self,
            from: Data(kKimiK25ConfigJSON.utf8)
        )

        #expect(config.modelType == "kimi_k25")
        #expect(config.textConfig.modelType == "kimi_k2")
        #expect(config.textConfig.hiddenSize == 7_168)
        #expect(config.textConfig.numHiddenLayers == 61)
        #expect(config.textConfig.numAttentionHeads == 64)
        #expect(config.textConfig.kvLoraRank == 512)
        #expect(config.textConfig.qLoraRank == 1_536)
        #expect(config.textConfig.nRoutedExperts == 384)
        #expect(config.textConfig.numExpertsPerTok == 8)
        #expect(config.textConfig.ropeScaling?["type"] == .string("yarn"))
        #expect(config.textConfig.ropeScaling?["factor"]?.asFloat() == 64)
    }

    @Test("registers and constructs Kimi K2.5 through the factory")
    func registersAndConstructsKimiK25ThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("kimi_k25"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("KimiK25ArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kKimiK25TinyFactoryConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "kimi_k25"
        )

        #expect(model is KimiK25Model)
        #expect((model as? KimiK25Model)?.vocabularySize == 64)
    }

    @Test("tiny Kimi K2.5 model produces finite logits")
    func tinyKimiK25ModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = KimiK25Model(.init(textConfig: Self.smallTextConfig()))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
            #expect(model.newCache(parameters: nil).count == 2)
            #expect(model.loraLinearLayers().count == 2)
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = KimiK25Model(.init(textConfig: Self.smallTextConfig()))
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("sanitizer unwraps language weights and prepares MLA projections")
    func sanitizerUnwrapsLanguageWeightsAndPreparesMLAProjections() throws {
        Device.withDefaultDevice(.cpu) {
            let model = KimiK25Model(.init(textConfig: Self.smallTextConfig()))
            let sanitized = model.sanitize(weights: [
                "vision_tower.blocks.0.weight": MLXArray.ones([1]),
                "language_model.lm_head.weight": MLXArray.ones([64, 16]),
                "language_model.model.layers.0.self_attn.kv_b_proj.weight": MLXArray.ones([16, 6]),
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2])
            ])

            let embedQ = sanitized["language_model.model.layers.0.self_attn.embed_q.weight"]
            let unembedOut = sanitized["language_model.model.layers.0.self_attn.unembed_out.weight"]

            #expect(sanitized["vision_tower.blocks.0.weight"] == nil)
            #expect(sanitized["language_model.model.layers.0.self_attn.kv_b_proj.weight"] == nil)
            #expect(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["language_model.lm_head.weight"]?.shape == [64, 16])
            #expect(embedQ?.shape == [2, 6, 4])
            #expect(unembedOut?.shape == [2, 4, 6])
        }
    }

    private static func smallTextConfig() -> DeepseekV3Configuration {
        DeepseekV3Configuration(
            modelType: "kimi_k2",
            vocabSize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            nSharedExperts: nil,
            nRoutedExperts: nil,
            routedScalingFactor: 1.5,
            kvLoraRank: 6,
            qLoraRank: 5,
            qkRopeHeadDim: 4,
            vHeadDim: 4,
            qkNopeHeadDim: 4,
            normTopkProb: true,
            nGroup: nil,
            topkGroup: nil,
            numExpertsPerTok: nil,
            moeLayerFreq: 1,
            firstKDenseReplace: 1,
            maxPositionEmbeddings: 64,
            rmsNormEps: 1e-5,
            ropeTheta: 10_000,
            attentionBias: false
        )
    }
}

private let kKimiK25TinyFactoryConfigJSON = #"""
{
    "model_type": "kimi_k25",
    "text_config": {
        "model_type": "kimi_k2",
        "hidden_size": 16,
        "intermediate_size": 32,
        "kv_lora_rank": 6,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "q_lora_rank": 5,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "v_head_dim": 4,
        "vocab_size": 64
    }
}
"""#

private let kKimiK25ConfigJSON = #"""
{
    "architectures": ["KimiK25ForConditionalGeneration"],
    "model_type": "kimi_k25",
    "text_config": {
        "model_type": "kimi_k2",
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 61,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
        "kv_lora_rank": 512,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 384,
        "n_shared_experts": 1,
        "routed_scaling_factor": 2.827,
        "num_experts_per_tok": 8,
        "moe_layer_freq": 1,
        "first_k_dense_replace": 1,
        "norm_topk_prob": true,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "scoring_func": "sigmoid",
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "rope_scaling": {
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "factor": 64.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn"
        },
        "tie_word_embeddings": false,
        "vocab_size": 163840
    }
}
"""#
