import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Phi architecture")
struct PhiArchitectureTests {
    @Test("decodes Phi configuration with project defaults")
    func decodesPhiConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 16,
            "intermediate_size": 32,
            "partial_rotary_factor": 0.5,
            "layer_norm_eps": 0.00001
        }
        """#

        let config = try JSONDecoder.json5().decode(PhiConfiguration.self, from: Data(json.utf8))

        #expect(config.hiddenSize == 8)
        #expect(config.attentionHeads == 2)
        #expect(config.hiddenLayers == 1)
        #expect(config.vocabularySize == 16)
        #expect(config.intermediateSize == 32)
        #expect(config.kvHeads == 2)
        #expect(config.maxPositionalEmbeddings == 2_048)
        #expect(config.ropeTheta == 10_000)
    }

    @Test("builds explicit Phi attention layout")
    func buildsExplicitPhiAttentionLayout() {
        let config = PhiConfiguration(
            vocabularySize: 32,
            hiddenSize: 16,
            attentionHeads: 4,
            hiddenLayers: 1,
            kvHeads: 2,
            partialRotaryFactor: 0.5,
            intermediateSize: 32
        )

        let layout = PhiAttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.rotaryDimensions == 2)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("clamps rotary dimensions to the head size")
    func clampsRotaryDimensionsToHeadSize() {
        let config = PhiConfiguration(
            vocabularySize: 32,
            hiddenSize: 16,
            attentionHeads: 4,
            hiddenLayers: 1,
            partialRotaryFactor: 1.5,
            intermediateSize: 32
        )

        let layout = PhiAttentionLayout(config)

        #expect(layout.headSize == 4)
        #expect(layout.rotaryDimensions == 4)
    }

    @Test("constructs Phi model with greedy fast path and LoRA layers")
    func constructsPhiModelWithGreedyFastPathAndLoRALayers() {
        let config = PhiConfiguration(
            vocabularySize: 32,
            hiddenSize: 16,
            attentionHeads: 4,
            hiddenLayers: 1,
            kvHeads: 2,
            partialRotaryFactor: 0.5,
            intermediateSize: 32
        )

        let model = PhiModel(config)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 32)
        #expect(model.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }
}
