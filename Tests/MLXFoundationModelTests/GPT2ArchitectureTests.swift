import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GPT-2 architecture")
struct GPT2ArchitectureTests {
    @Test("decodes GPT-2 configuration with defaults")
    func decodesConfigurationWithDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            GPT2Configuration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "gpt2")
        #expect(config.contextLength == config.maxPositionEmbeddings)
        #expect(config.feedForwardSize == 64)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.activationFunction == "gelu_new")
    }

    @Test("builds attention layout")
    func buildsAttentionLayout() {
        let layout = GPT2AttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.attentionHeads == 4)
        #expect(layout.headDimensions == 4)
        #expect(layout.combinedProjectionSize == 48)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = GPT2Model(Self.smallConfig(hiddenLayers: 2))
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
            let model = GPT2Model(Self.smallConfig())
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

    @Test("sanitizer accepts raw Transformers keys")
    func sanitizerAcceptsRawTransformersKeys() throws {
        let model = GPT2Model(Self.smallConfig(hiddenSize: 4, attentionHeads: 2, hiddenLayers: 1))
        let sanitized = model.sanitize(weights: Self.rawTransformersWeights())

        #expect(sanitized["model.h.0.attn.bias"] == nil)
        #expect(sanitized["model.h.0.attn.masked_bias"] == nil)
        #expect(sanitized["model.lm_head.weight"] == nil)
        #expect(sanitized["model.wte.weight"] != nil)

        let attentionQKV = try #require(sanitized["model.h.0.attn.c_attn.weight"])
        let attentionOutput = try #require(sanitized["model.h.0.attn.c_proj.weight"])
        let feedForwardUp = try #require(sanitized["model.h.0.mlp.c_fc.weight"])

        eval(attentionQKV, attentionOutput, feedForwardUp)
        #expect(attentionQKV.shape == [12, 4])
        #expect(attentionOutput.shape == [4, 4])
        #expect(feedForwardUp.shape == [16, 4])
        #expect(Array(attentionOutput.asArray(Float.self).prefix(4)) == [1, 5, 9, 13])
    }

    @Test("sanitizer preserves prefixed MLX keys")
    func sanitizerPreservesPrefixedMLXKeys() throws {
        let model = GPT2Model(Self.smallConfig(hiddenSize: 4, attentionHeads: 2, hiddenLayers: 1))
        let prefixedWeight = MLXArray((0 ..< 16).map(Float.init)).reshaped([4, 4])
        let sanitized = model.sanitize(weights: [
            "model.h.0.attn.c_proj.weight": prefixedWeight,
            "model.h.0.attn.bias": MLXArray.ones([1])
        ])

        #expect(sanitized["model.h.0.attn.bias"] == nil)
        let preserved = try #require(sanitized["model.h.0.attn.c_proj.weight"])
        eval(preserved)
        #expect(preserved.shape == [4, 4])
        #expect(Array(preserved.asArray(Float.self).prefix(4)) == [0, 1, 2, 3])
    }

    private static func smallConfig(
        hiddenSize: Int = 16,
        attentionHeads: Int = 4,
        hiddenLayers: Int = 1
    ) -> GPT2Configuration {
        GPT2Configuration(
            contextLength: 16,
            hiddenSize: hiddenSize,
            attentionHeads: attentionHeads,
            hiddenLayers: hiddenLayers,
            maxPositionEmbeddings: 16,
            vocabularySize: 64
        )
    }

    private static func rawTransformersWeights() -> [String: MLXArray] {
        [
            "transformer.wte.weight": MLXArray.ones([64, 4]),
            "transformer.h.0.attn.bias": MLXArray.ones([1]),
            "transformer.h.0.attn.masked_bias": MLXArray.ones([1]),
            "transformer.h.0.attn.c_attn.weight": MLXArray((0 ..< 48).map(Float.init))
                .reshaped([4, 12]),
            "transformer.h.0.attn.c_proj.weight": MLXArray((1 ... 16).map(Float.init))
                .reshaped([4, 4]),
            "transformer.h.0.mlp.c_fc.weight": MLXArray((0 ..< 64).map(Float.init))
                .reshaped([4, 16]),
            "lm_head.weight": MLXArray.ones([64, 4])
        ]
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "gpt2",
            "n_embd": 16,
            "n_head": 4,
            "n_layer": 1,
            "n_positions": 16,
            \(includeOptionalFields ? "\"n_ctx\": 32," : "")
            \(includeOptionalFields ? "\"n_inner\": 48," : "")
            "vocab_size": 64
        }
        """
    }
}
