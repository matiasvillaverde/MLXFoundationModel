import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("OLMo2 architecture")
struct Olmo2ArchitectureTests {
    @Test("decodes OLMo2 configuration with project defaults")
    func decodesOlmo2ConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 128
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Olmo2Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.ropeTraditional == false)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.attentionBias == false)
        #expect(config.mlpBias == false)
    }

    @Test("builds OLMo2 attention layout")
    func buildsOlmo2AttentionLayout() {
        let layout = Olmo2AttentionLayout(Self.smallConfig(headDimensions: 8))

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 8)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
        #expect(layout.attentionScale == 1 / Float(8).squareRoot())
    }

    @Test("constructs OLMo2 model with cache, adapters, and greedy fast path")
    func constructsOlmo2ModelWithCacheAdaptersAndGreedyFastPath() throws {
        let model = Olmo2Model(Self.smallConfig(hiddenLayers: 2))
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

    @Test("tiny OLMo2 model produces finite logits")
    func tinyOlmo2ModelProducesFiniteLogits() {
        let model = Olmo2Model(Self.smallConfig(hiddenLayers: 1))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer removes tied output head and rotary checkpoint tensors")
    func sanitizerRemovesTiedOutputHeadAndRotaryCheckpointTensors() {
        let model = Olmo2Model(Self.smallConfig(tieWordEmbeddings: true))
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
        headDimensions: Int? = nil,
        tieWordEmbeddings: Bool = true
    ) -> Olmo2Configuration {
        Olmo2Configuration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            headDimensions: headDimensions,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 64,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
