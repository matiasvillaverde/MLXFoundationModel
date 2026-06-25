import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MiniCPM architecture")
struct MiniCPMArchitectureTests {
    @Test("decodes MiniCPM configuration with project defaults")
    func decodesMiniCPMConfigurationWithDefaults() throws {
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
            MiniCPMConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "minicpm")
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.dimModelBase == 16)
        #expect(config.scaleDepth == 1)
        #expect(config.scaleEmb == 1)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("builds MiniCPM attention layout and scale plan")
    func buildsMiniCPMAttentionLayoutAndScalePlan() {
        let config = Self.smallConfig(
            hiddenLayers: 4,
            dimModelBase: 8,
            scaleDepth: 2,
            scaleEmb: 1.5
        )

        let layout = MiniCPMAttentionLayout(config)
        let scalePlan = MiniCPMScalePlan(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
        #expect(scalePlan.embeddingScale == 1.5)
        #expect(scalePlan.residualScale == 1)
        #expect(scalePlan.logitDivisor == 2)
    }

    @Test("constructs MiniCPM model with cache, adapters, and greedy fast path")
    func constructsMiniCPMModelWithCacheAdaptersAndGreedyFastPath() {
        let model = MiniCPMModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny MiniCPM model produces finite logits")
    func tinyMiniCPMModelProducesFiniteLogits() {
        let model = MiniCPMModel(Self.smallConfig(hiddenLayers: 1))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer removes tied output head and rotary checkpoint tensors")
    func sanitizerRemovesTiedOutputHeadAndRotaryCheckpointTensors() {
        let model = MiniCPMModel(Self.smallConfig(tieWordEmbeddings: true))
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
    ) -> MiniCPMConfiguration {
        MiniCPMConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 64,
            dimModelBase: dimModelBase,
            scaleDepth: scaleDepth,
            scaleEmb: scaleEmb,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
