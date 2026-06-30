import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("OLMo architecture")
struct OlmoArchitectureTests {
    @Test("decodes HF OLMo configuration")
    func decodesHFOlmoConfiguration() throws {
        let json = #"""
        {
            "model_type": "olmo",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 64,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "tie_word_embeddings": true,
            "clip_qkv": 8.0
        }
        """#

        let config = try JSONDecoder.json5().decode(
            OlmoConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "olmo")
        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 2)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 2)
        #expect(config.embeddingSize == 64)
        #expect(config.layerNormEps == 1e-5)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.clipQKV == 8)
    }

    @Test("decodes legacy mlx-lm OLMo configuration")
    func decodesLegacyMLXLMOlmoConfiguration() throws {
        let json = #"""
        {
            "model_type": "olmo",
            "d_model": 16,
            "n_layers": 2,
            "mlp_hidden_size": 64,
            "n_heads": 4,
            "vocab_size": 60,
            "embedding_size": 64,
            "weight_tying": false,
            "rope_traditional": false
        }
        """#

        let config = try JSONDecoder.json5().decode(
            OlmoConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 2)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.embeddingSize == 64)
        #expect(config.kvHeads == 4)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("builds OLMo attention layout")
    func buildsOlmoAttentionLayout() {
        let layout = OlmoAttentionLayout(Self.smallConfig(headDimensions: 8))

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 8)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
        #expect(layout.attentionScale == 1 / Float(8).squareRoot())
    }

    @Test("constructs OLMo model with cache, adapters, and greedy fast path")
    func constructsOlmoModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = OlmoModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        _ = try #require(cache[0] as? KVCacheSimple)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny tied OLMo model produces finite logits")
    func tinyTiedOlmoModelProducesFiniteLogits() {
        let model = OlmoModel(Self.smallConfig(hiddenLayers: 1, tieWordEmbeddings: true))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("tiny untied OLMo model uses output head")
    func tinyUntiedOlmoModelUsesOutputHead() {
        let model = OlmoModel(Self.smallConfig(hiddenLayers: 1, tieWordEmbeddings: false))
        let logits = model(MLXArray([1, 2]).reshaped(1, 2), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 2, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer drops tied head, rotary, and affine norm leftovers")
    func sanitizerDropsTiedHeadRotaryAndAffineNormLeftovers() {
        let model = OlmoModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.0.input_layernorm.weight": MLXArray.ones([16]),
            "model.layers.0.post_attention_layernorm.bias": MLXArray.zeros([16]),
            "model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.layers.0.input_layernorm.weight"] == nil)
        #expect(sanitized["model.layers.0.post_attention_layernorm.bias"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
    }

    @Test("sanitizer maps legacy packed OLMo keys")
    func sanitizerMapsLegacyPackedOlmoKeys() {
        let model = OlmoModel(Self.smallConfig(tieWordEmbeddings: false))
        let sanitized = model.sanitize(weights: [
            "model.transformer.wte.weight": MLXArray.ones([64, 16]),
            "model.transformer.blocks.0.att_proj.weight": MLXArray.ones([48, 16]),
            "model.transformer.blocks.0.attn_out.weight": MLXArray.ones([16, 16]),
            "model.transformer.blocks.0.ff_proj.weight": MLXArray.ones([64, 16]),
            "model.transformer.blocks.0.ff_out.weight": MLXArray.ones([16, 32]),
            "model.lm_head.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["model.embed_tokens.weight"]?.shape == [64, 16])
        #expect(sanitized["model.layers.0.self_attn.q_proj.weight"]?.shape == [16, 16])
        #expect(sanitized["model.layers.0.self_attn.k_proj.weight"]?.shape == [16, 16])
        #expect(sanitized["model.layers.0.self_attn.v_proj.weight"]?.shape == [16, 16])
        #expect(sanitized["model.layers.0.self_attn.o_proj.weight"]?.shape == [16, 16])
        #expect(sanitized["model.layers.0.mlp.up_proj.weight"]?.shape == [32, 16])
        #expect(sanitized["model.layers.0.mlp.gate_proj.weight"]?.shape == [32, 16])
        #expect(sanitized["model.layers.0.mlp.down_proj.weight"]?.shape == [16, 32])
        #expect(sanitized["lm_head.weight"]?.shape == [64, 16])
        #expect(sanitized["model.lm_head.weight"] == nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        headDimensions: Int? = nil,
        tieWordEmbeddings: Bool = true
    ) -> OlmoConfiguration {
        OlmoConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            headDimensions: headDimensions,
            layerNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
