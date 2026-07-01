import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Youtu LLM architecture")
struct YoutuLLMArchitectureTests {
    @Test("decodes Youtu MLA configuration")
    func decodesYoutuMLAConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            YoutuLLMConfiguration.self,
            from: Data(Self.tencentConfigJSON.utf8)
        )
        let layout = DeepseekV3AttentionLayout(config)

        #expect(config.modelType == "youtu")
        #expect(config.hiddenSize == 2_048)
        #expect(config.intermediateSize == 6_144)
        #expect(config.numAttentionHeads == 16)
        #expect(config.numKeyValueHeads == 16)
        #expect(config.qLoraRank == 1_536)
        #expect(config.kvLoraRank == 512)
        #expect(config.qkNopeHeadDim == 128)
        #expect(config.qkRopeHeadDim == 64)
        #expect(config.vHeadDim == 128)
        #expect(config.ropeTheta == 1_600_000)
        #expect(config.tieWordEmbeddings)
        #expect(layout.queryHeadSize == 192)
        #expect(layout.compressedKeyValueSize == 576)
        #expect(layout.keyValueProjectionSize == 4_096)
        #expect(layout.outputProjectionSize == 2_048)
    }

    @Test("registers Youtu model aliases")
    func registersYoutuModelAliases() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.contains("youtu"))
        #expect(registeredTypes.contains("youtu_llm"))
    }

    @Test("constructs Youtu model through the factory")
    func constructsYoutuModelThroughFactory() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("YoutuLLMArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try Self.factoryConfigJSON.write(to: configurationURL, atomically: true, encoding: .utf8)

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "youtu_llm"
        )

        #expect(model is YoutuLLMModel)
        #expect((model as? YoutuLLMModel)?.vocabularySize == 64)
    }

    @Test("tiny tied Youtu model produces finite logits")
    func tinyTiedYoutuModelProducesFiniteLogits() {
        let model = YoutuLLMModel(Self.smallConfig(tieWordEmbeddings: true))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer drops tied output head checkpoint tensors")
    func sanitizerDropsTiedOutputHeadCheckpointTensors() {
        let model = YoutuLLMModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([64, 1]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static let tencentConfigJSON = #"""
    {
        "model_type": "youtu",
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "kv_lora_rank": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 32,
        "num_key_value_heads": 16,
        "q_lora_rank": 1536,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "tie_word_embeddings": true,
        "v_head_dim": 128,
        "vocab_size": 128256,
        "rope_parameters": {
            "rope_theta": 1600000,
            "rope_type": "default"
        }
    }
    """#

    private static let factoryConfigJSON = #"""
    {
        "model_type": "youtu_llm",
        "hidden_size": 16,
        "intermediate_size": 32,
        "kv_lora_rank": 6,
        "num_attention_heads": 2,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 8,
        "qk_nope_head_dim": 4,
        "qk_rope_head_dim": 4,
        "tie_word_embeddings": true,
        "v_head_dim": 4,
        "vocab_size": 64,
        "rope_parameters": {
            "rope_theta": 10000
        }
    }
    """#

    private static func smallConfig(tieWordEmbeddings: Bool) -> YoutuLLMConfiguration {
        YoutuLLMConfiguration(
            modelType: "youtu_llm",
            vocabSize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            moeIntermediateSize: 32,
            numHiddenLayers: 2,
            numAttentionHeads: 2,
            numKeyValueHeads: 2,
            kvLoraRank: 6,
            qLoraRank: 8,
            qkRopeHeadDim: 4,
            vHeadDim: 4,
            qkNopeHeadDim: 4,
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
