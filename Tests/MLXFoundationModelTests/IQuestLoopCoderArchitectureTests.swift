import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("IQuest Loop Coder architecture")
struct IQuestLoopCoderArchitectureTests {
    @Test("decodes IQuest Loop Coder checkpoint configuration")
    func decodesIQuestLoopCoderCheckpointConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            IQuestLoopCoderConfiguration.self,
            from: Data(kIQuestLoopCoderConfigJSON.utf8)
        )

        #expect(config.modelType == "iquestloopcoder")
        #expect(config.hiddenSize == 5_120)
        #expect(config.hiddenLayers == 80)
        #expect(config.intermediateSize == 27_648)
        #expect(config.attentionHeads == 40)
        #expect(config.keyValueHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.loopCount == 2)
        #expect(config.loopWindowSize == 64)
        #expect(config.vocabularySize == 76_800)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("registers and constructs IQuest Loop Coder through the factory")
    func registersAndConstructsIQuestLoopCoderThroughFactory() throws {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())
        #expect(registeredTypes.contains("iquestloopcoder"))

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("IQuestLoopCoderArchitectureTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configurationURL = directory.appendingPathComponent("config.json")
        try kIQuestLoopCoderTinyConfigJSON.write(
            to: configurationURL,
            atomically: true,
            encoding: .utf8
        )

        let model = try LLMTypeRegistry.shared.createModel(
            configuration: configurationURL,
            modelType: "iquestloopcoder"
        )

        #expect(model is IQuestLoopCoderModel)
        #expect((model as? IQuestLoopCoderModel)?.vocabularySize == 64)
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = IQuestLoopCoderModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path accepts unbatched text input")
    func greedyTokenPathAcceptsUnbatchedTextInput() {
        Device.withDefaultDevice(.cpu) {
            let model = IQuestLoopCoderModel(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("creates full and local loop caches")
    func createsFullAndLocalLoopCaches() throws {
        let model = IQuestLoopCoderModel(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()

        #expect(model.kvHeads == [1, 1, 1, 1])
        #expect(cache.count == 4)
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        _ = try #require(cache[2] as? RotatingKVCache)
        _ = try #require(cache[3] as? RotatingKVCache)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "k_proj", "v_proj", "o_proj"])
    }

    @Test("sanitizer removes stale rotary and tied-head weights")
    func sanitizerRemovesStaleRotaryAndTiedHeadWeights() {
        let tiedModel = IQuestLoopCoderModel(Self.smallConfig(tieWordEmbeddings: true))
        let untiedModel = IQuestLoopCoderModel(Self.smallConfig(tieWordEmbeddings: false))
        let weights: [String: MLXArray] = [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ]

        let tiedSanitized = tiedModel.sanitize(weights: weights)
        let untiedSanitized = untiedModel.sanitize(weights: weights)

        #expect(tiedSanitized["lm_head.weight"] == nil)
        #expect(tiedSanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(tiedSanitized["model.embed_tokens.weight"]?.shape == [64, 16])
        #expect(untiedSanitized["lm_head.weight"]?.shape == [64, 16])
    }

    private static func smallConfig(
        tieWordEmbeddings: Bool = false
    ) -> IQuestLoopCoderConfiguration {
        IQuestLoopCoderConfiguration(
            hiddenSize: 16,
            hiddenLayers: 2,
            intermediateSize: 32,
            attentionHeads: 2,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            headDim: 8,
            keyValueHeads: 1,
            maxPositionEmbeddings: 64,
            attentionBias: false,
            mlpBias: false,
            ropeTheta: 10_000,
            tieWordEmbeddings: tieWordEmbeddings,
            loopCount: 2,
            loopWindowSize: 4
        )
    }
}

private let kIQuestLoopCoderTinyConfigJSON = #"""
{
    "model_type": "iquestloopcoder",
    "hidden_size": 16,
    "num_hidden_layers": 2,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 64,
    "head_dim": 8,
    "num_key_value_heads": 1,
    "max_position_embeddings": 64,
    "attention_bias": false,
    "mlp_bias": false,
    "rope_theta": 10000,
    "tie_word_embeddings": false,
    "loop_num": 2,
    "loop_window_size": 4
}
"""#

private let kIQuestLoopCoderConfigJSON = #"""
{
    "architectures": ["IQuestLoopCoderForCausalLM"],
    "model_type": "iquestloopcoder",
    "hidden_size": 5120,
    "num_hidden_layers": 80,
    "intermediate_size": 27648,
    "num_attention_heads": 40,
    "rms_norm_eps": 1e-6,
    "vocab_size": 76800,
    "head_dim": 128,
    "num_key_value_heads": 8,
    "max_position_embeddings": 131072,
    "attention_bias": false,
    "mlp_bias": false,
    "rope_theta": 500000,
    "tie_word_embeddings": false,
    "loop_num": 2,
    "loop_window_size": 64
}
"""#
