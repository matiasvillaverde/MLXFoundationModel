import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("RWKV7 architecture")
struct RWKV7ArchitectureTests {
    @Test("decodes tiny Goose checkpoint configuration")
    func decodesTinyGooseCheckpointConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            RWKV7Configuration.self,
            from: Data(Self.tinyGooseConfigJSON.utf8)
        )

        #expect(config.modelType == "rwkv7")
        #expect(config.vocabularySize == 65_536)
        #expect(config.hiddenSize == 768)
        #expect(config.intermediateSize == 3_072)
        #expect(config.hiddenLayers == 12)
        #expect(config.headSize == 64)
        #expect(config.headCount == 12)
        #expect(config.layerNormEpsilon == 1e-5)
        #expect(config.groupNormEpsilon == 64e-5)
        #expect(config.inContextLearningRank == 64)
        #expect(config.valueRank == 32)
        #expect(config.gateRank == 128)
        #expect(config.decayRank == 64)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("plans WKV recurrence layout")
    func plansWKVRecurrenceLayout() {
        let tensor = MLXArray.zeros([2, 3, 4, 32])
        let layout = RWKV7WKVLayout(receptance: tensor, key: tensor, value: tensor)

        #expect(layout.batchSize == 2)
        #expect(layout.sequenceLength == 3)
        #expect(layout.headCount == 4)
        #expect(layout.headSize == 32)
        #expect(layout.stateShape == [2, 4, 32, 32])
        #expect(layout.supportsMetalKernel)
    }

    @Test("WKV recurrence returns output and next state")
    func wkvRecurrenceReturnsOutputAndNextState() {
        Device.withDefaultDevice(.cpu) {
            let shape = [1, 2, 2, 4]
            let receptance = MLXArray.ones(shape)
            let decay = MLXArray.ones(shape) * 0.5
            let key = MLXArray.ones(shape) * 0.25
            let value = MLXArray.ones(shape) * 0.75
            let stateKey = MLXArray.ones(shape) * 0.1
            let stateValue = MLXArray.ones(shape) * -0.2

            let (output, state) = rwkv7WKVUpdate(
                receptance: receptance,
                decay: decay,
                key: key,
                value: value,
                stateKey: stateKey,
                stateValue: stateValue,
                state: nil
            )
            eval(output, state)

            #expect(output.shape == shape)
            #expect(state.shape == [1, 2, 4, 4])
            #expect(all(isFinite(output)).item(Bool.self))
            #expect(all(isFinite(state)).item(Bool.self))
        }
    }

    @Test("constructs model with cache lists and greedy fast path")
    func constructsModelWithCacheListsAndGreedyFastPath() throws {
        let model = RWKV7Model(Self.smallConfig())
        let cache = model.newCache(parameters: nil)
        let _: any GreedyTokenModel = model

        #expect(cache.count == 2)
        let first = try #require(cache[0] as? CacheList)
        #expect(first.layoutCaches.count == 2)
        _ = try #require(first[0] as? MambaCache)
        _ = try #require(first[1] as? MambaCache)
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = RWKV7Model(Self.smallConfig(hiddenLayers: 1))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    private static func smallConfig(hiddenLayers: Int = 2) -> RWKV7Configuration {
        RWKV7Configuration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            hiddenLayers: hiddenLayers,
            headSize: 4,
            inContextLearningRank: 4,
            valueRank: 4,
            gateRank: 4,
            decayRank: 4
        )
    }

    private static let tinyGooseConfigJSON = #"""
    {
        "architectures": ["Rwkv7ForCausalLM"],
        "bos_token_id": 0,
        "context_length": 8192,
        "decay_lora_rank": 64,
        "eos_token_id": 0,
        "gate_lora_rank": 128,
        "group_norm_epsilon": 0.00064,
        "head_size": 64,
        "hidden_size": 768,
        "in_context_learning_lora_rank": 64,
        "intermediate_size": 3072,
        "layer_norm_epsilon": 0.00001,
        "model_type": "rwkv7",
        "num_hidden_layers": 12,
        "tie_word_embeddings": false,
        "use_cache": true,
        "value_lora_rank": 32,
        "vocab_size": 65536
    }
    """#
}
