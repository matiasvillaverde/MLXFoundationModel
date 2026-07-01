import Foundation
import MLX
import MLXFast
@testable import MLXLocalModels
import Testing

@Suite("Nemotron architecture")
struct NemotronArchitectureTests {
    @Test("decodes configuration with Nemotron defaults")
    func decodesConfigurationWithNemotronDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: false).utf8)
        )

        #expect(config.modelType == "nemotron")
        #expect(config.hiddenActivation == "relu2")
        #expect(config.kvHeads == config.attentionHeads)
        #expect(config.resolvedHeadDimensions == 4)
        #expect(config.normEps == 1e-5)
        #expect(config.partialRotaryFactor == 0.5)
        #expect(config.ropeTheta == 10_000)
        #expect(!config.tieWordEmbeddings)
    }

    @Test("builds attention layout with partial linear RoPE")
    func buildsAttentionLayoutWithPartialLinearRoPE() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronConfiguration.self,
            from: Data(Self.configJSON(includeOptionalFields: true).utf8)
        )
        let layout = NemotronAttentionLayout(config)

        #expect(config.kvHeads == 2)
        #expect(config.headDimensions == 8)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDimensions == 8)
        #expect(layout.rotaryDimensions == 2)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
        #expect(layout.attentionScale == Float(1 / sqrt(8.0)))
        #expect(layout.ropeScale == 0.25)
    }

    @Test("LayerNorm1P applies checkpoint weight plus one")
    func layerNorm1PAppliesCheckpointWeightPlusOne() {
        Device.withDefaultDevice(.cpu) {
            let norm = NemotronLayerNorm1P(dimensions: 4, eps: 1e-5)
            let input = MLXArray((0 ..< 8).map(Float.init)).reshaped(2, 4)
            let expected = MLXFast.layerNorm(
                input,
                weight: MLXArray.ones([4]) * 2,
                bias: MLXArray.zeros([4]),
                eps: 1e-5
            )
            let output = norm(input)
            eval(output, expected)

            #expect(all(isClose(output, expected)).item(Bool.self))
        }
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = NemotronModel(Self.smallConfig(hiddenLayers: 2, kvHeads: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "nemotron")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = NemotronModel(Self.smallConfig(hiddenLayers: 2))
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

    @Test("tiny tied model produces finite logits")
    func tinyTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = NemotronModel(Self.smallConfig(tieWordEmbeddings: true))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer strips rotary metadata and tied output head")
    func sanitizerStripsRotaryMetadataAndTiedOutputHead() {
        let model = NemotronModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([1]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        kvHeads: Int? = nil,
        tieWordEmbeddings: Bool = false
    ) -> NemotronConfiguration {
        NemotronConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            kvHeads: kvHeads,
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func configJSON(includeOptionalFields: Bool) -> String {
        """
        {
            "model_type": "nemotron",
            "hidden_size": 16,
            "hidden_act": "relu2",
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            \(includeOptionalFields ? "\"num_key_value_heads\": 2," : "")
            \(includeOptionalFields ? "\"head_dim\": 8," : "")
            "norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 64,
            "partial_rotary_factor": \(includeOptionalFields ? "0.25" : "0.5"),
            "rope_theta": 10000,
            \(includeOptionalFields ? "\"rope_scaling\": {\"type\": \"linear\", \"factor\": 4}," : "")
            "tie_word_embeddings": false
        }
        """
    }
}
