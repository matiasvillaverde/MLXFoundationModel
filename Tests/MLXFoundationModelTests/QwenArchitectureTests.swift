import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Qwen architecture")
struct QwenArchitectureTests {
    @Test("decodes Qwen configuration with checkpoint defaults")
    func decodesConfigurationWithCheckpointDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            QwenConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "qwen")
        #expect(config.hiddenSize == 2_048)
        #expect(config.kvChannels == 128)
        #expect(config.keyValueHeads == 16)
        #expect(config.ropeBase == 10_000)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention layout")
    func buildsAttentionLayout() {
        let layout = QwenAttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.headDim == 4)
        #expect(layout.projectionSize == 16)
        #expect(layout.combinedProjectionSize == 48)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = QwenModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [4, 4])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["c_attn", "c_proj"])
    }

    @Test("tiny model produces finite logits with and without cache")
    func tinyModelProducesFiniteLogitsWithAndWithoutCache() {
        Device.withDefaultDevice(.cpu) {
            let model = QwenModel(Self.smallConfig())
            let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
            let logits = model(tokens, cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))

            let cache = model.newCache(parameters: nil)
            let cachedPrefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let cachedNext = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(cachedPrefill, cachedNext)

            #expect(cachedPrefill.shape == [1, 2, 64])
            #expect(cachedNext.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(cachedNext)).item(Bool.self))
        }
    }

    @Test("sanitizer strips unused tensors for tied heads")
    func sanitizerStripsUnusedTensorsForTiedHeads() {
        let model = QwenModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "transformer.h.0.attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "transformer.wte.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["transformer.h.0.attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["transformer.wte.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        tieWordEmbeddings: Bool = false
    ) -> QwenConfiguration {
        QwenConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static let configJSON = """
    {
        "model_type": "qwen",
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "kv_channels": 128,
        "layer_norm_epsilon": 0.000001,
        "max_position_embeddings": 8192,
        "no_bias": true,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "rotary_emb_base": 10000,
        "tie_word_embeddings": false,
        "vocab_size": 151936
    }
    """
}
