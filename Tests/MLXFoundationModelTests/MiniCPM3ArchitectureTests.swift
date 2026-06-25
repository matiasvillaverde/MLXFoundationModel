import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MiniCPM3 architecture")
struct MiniCPM3ArchitectureTests {
    @Test("decodes MiniCPM3 configuration with defaults")
    func decodesMiniCPM3ConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPM3Configuration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "minicpm3")
        #expect(config.dimModelBase == config.hiddenSize)
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.ropeTheta == 1_000_000)
        #expect(!config.ropeTraditional)
        #expect(!config.attentionBias)
        #expect(!config.tieWordEmbeddings)
        #expect(config.scaleDepth == 1)
        #expect(config.scaleEmb == 1)
    }

    @Test("builds MLA attention layout and scale plan")
    func buildsMLAAttentionLayoutAndScalePlan() {
        let config = Self.smallConfig(hiddenLayers: 4, dimModelBase: 8, scaleDepth: 2)
        let layout = MiniCPM3AttentionLayout(config)
        let scalePlan = MiniCPM3ScalePlan(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.queryLowRank == 8)
        #expect(layout.keyValueLowRank == 6)
        #expect(layout.nopeHeadSize == 2)
        #expect(layout.ropeHeadSize == 2)
        #expect(layout.valueHeadSize == 4)
        #expect(layout.queryHeadSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.compressedKeyValueSize == 8)
        #expect(layout.keyValueProjectionSize == 24)
        #expect(layout.outputProjectionSize == 16)
        #expect(layout.attentionScale == 0.5)
        #expect(scalePlan.residualScale == 1)
        #expect(scalePlan.logitDivisor == 2)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = MiniCPM3Model(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [4, 4])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"])
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = MiniCPM3Model(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer removes tied output head and rotary checkpoint tensors")
    func sanitizerRemovesTiedOutputHeadAndRotaryCheckpointTensors() {
        let model = MiniCPM3Model(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        dimModelBase: Int? = nil,
        scaleDepth: Float = 1,
        scaleEmb: Float = 1,
        tieWordEmbeddings: Bool = false
    ) -> MiniCPM3Configuration {
        MiniCPM3Configuration(
            hiddenSize: 16,
            dimModelBase: dimModelBase,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            qLoraRank: 8,
            qkNopeHeadDim: 2,
            qkRopeHeadDim: 2,
            kvLoraRank: 6,
            scaleDepth: scaleDepth,
            scaleEmb: scaleEmb,
            maxPositionEmbeddings: 64,
            ropeScaling: [
                "type": .string("longrope"),
                "short_factor": .floats([1]),
                "long_factor": .floats([1]),
                "original_max_position_embeddings": .int(64)
            ],
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "minicpm3",
            "hidden_size": 16,
            \(includeOptionalFields ? "\"dim_model_base\": 8," : "")
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            \(includeOptionalFields ? "\"num_key_value_heads\": 4," : "")
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "q_lora_rank": 8,
            "qk_nope_head_dim": 2,
            "qk_rope_head_dim": 2,
            "kv_lora_rank": 6,
            "max_position_embeddings": 64
        }
        """
    }
}
