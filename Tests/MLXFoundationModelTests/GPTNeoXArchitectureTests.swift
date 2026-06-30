import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("GPT-NeoX architecture")
struct GPTNeoXArchitectureTests {
    @Test("decodes GPT-NeoX configuration with Pythia defaults")
    func decodesConfigurationWithPythiaDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            GPTNeoXConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "gpt_neox")
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.feedForwardSize == 64)
        #expect(config.rotaryEmbeddingBase == 10_000)
        #expect(config.rotaryPercent == 0.25)
        #expect(config.useParallelResidual)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention layout with partial RoPE and grouped KV")
    func buildsAttentionLayoutWithPartialRoPEAndGroupedKV() {
        let layout = GPTNeoXAttentionLayout(
            Self.smallConfig(attentionHeads: 4, kvHeads: 2, rotaryPercent: 0.5)
        )

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDimensions == 4)
        #expect(layout.rotaryDimensions == 2)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.combinedProjectionSize == 32)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = GPTNeoXModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["query_key_value", "dense"])
    }

    @Test("tiny parallel model produces finite logits with cache")
    func tinyParallelModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = GPTNeoXModel(Self.smallConfig(hiddenLayers: 2))
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("tiny sequential tied model produces finite logits")
    func tinySequentialTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = GPTNeoXModel(
                Self.smallConfig(useParallelResidual: false, tieWordEmbeddings: true)
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer maps raw Transformers keys")
    func sanitizerMapsRawTransformersKeys() throws {
        let model = GPTNeoXModel(
            Self.smallConfig(hiddenSize: 4, attentionHeads: 2, hiddenLayers: 1)
        )
        let sanitized = model.sanitize(weights: Self.rawTransformersWeights())

        #expect(sanitized["model.h.0.attention.bias"] == nil)
        #expect(sanitized["model.h.0.attention.masked_bias"] == nil)
        #expect(sanitized["model.h.0.attention.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_in.weight"] != nil)
        #expect(sanitized["model.embed_out.weight"] != nil)
        #expect(sanitized["model.h.0.attention.query_key_value.weight"] != nil)

        let qkv = try #require(sanitized["model.h.0.attention.query_key_value.weight"])
        eval(qkv)
        #expect(qkv.shape == [12, 4])
        #expect(Array(qkv.asArray(Float.self).prefix(4)) == [0, 1, 2, 3])
    }

    @Test("sanitizer strips untied head tensors when embeddings are tied")
    func sanitizerStripsUntiedHeadTensorsWhenEmbeddingsAreTied() {
        let model = GPTNeoXModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "model.embed_out.weight": MLXArray.ones([64, 16]),
            "model.embed_out.scales": MLXArray.ones([1]),
            "model.embed_out.biases": MLXArray.ones([1]),
            "model.embed_in.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.embed_out.weight"] == nil)
        #expect(sanitized["model.embed_out.scales"] == nil)
        #expect(sanitized["model.embed_out.biases"] == nil)
        #expect(sanitized["model.embed_in.weight"] != nil)
    }

    private static func smallConfig(
        hiddenSize: Int = 16,
        attentionHeads: Int = 4,
        hiddenLayers: Int = 1,
        kvHeads: Int? = nil,
        rotaryPercent: Float = 0.5,
        useParallelResidual: Bool = true,
        tieWordEmbeddings: Bool = false
    ) -> GPTNeoXConfiguration {
        GPTNeoXConfiguration(
            maxPositionEmbeddings: 64,
            hiddenSize: hiddenSize,
            attentionHeads: attentionHeads,
            hiddenLayers: hiddenLayers,
            vocabularySize: 64,
            rotaryPercent: rotaryPercent,
            useParallelResidual: useParallelResidual,
            kvHeads: kvHeads,
            intermediateSize: hiddenSize * 4,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func rawTransformersWeights() -> [String: MLXArray] {
        [
            "gpt_neox.embed_in.weight": MLXArray.ones([64, 4]),
            "gpt_neox.final_layer_norm.weight": MLXArray.ones([4]),
            "gpt_neox.layers.0.attention.bias": MLXArray.ones([1]),
            "gpt_neox.layers.0.attention.masked_bias": MLXArray.ones([1]),
            "gpt_neox.layers.0.attention.rotary_emb.inv_freq": MLXArray.ones([1]),
            "gpt_neox.layers.0.attention.query_key_value.weight": MLXArray(
                (0 ..< 48).map(Float.init)
            ).reshaped([12, 4]),
            "gpt_neox.layers.0.mlp.dense_h_to_4h.weight": MLXArray.ones([16, 4]),
            "embed_out.weight": MLXArray.ones([64, 4])
        ]
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "gpt_neox",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "num_hidden_layers": 1,
            \(includeOptionalFields ? "\"num_key_value_heads\": 2," : "")
            \(includeOptionalFields ? "\"use_parallel_residual\": false," : "")
            "intermediate_size": 64,
            "max_position_embeddings": 64,
            "rotary_emb_base": 10000,
            "rotary_pct": 0.25,
            "vocab_size": 64
        }
        """
    }
}
