import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Gemma architecture")
struct GemmaArchitectureTests {
    @Test("decodes Gemma configuration with project defaults")
    func decodesGemmaConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            GemmaConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "gemma")
        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.headDimensions == 4)
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.ropeTraditional == false)
    }

    @Test("builds explicit Gemma attention layout")
    func buildsExplicitGemmaAttentionLayout() {
        let config = Self.smallConfig()

        let layout = GemmaAttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("uses provided Gemma head dimensions")
    func usesProvidedGemmaHeadDimensions() {
        let config = GemmaConfiguration(
            hiddenSize: 24,
            hiddenLayers: 1,
            intermediateSize: 48,
            attentionHeads: 4,
            headDimensions: 8,
            vocabularySize: 64,
            kvHeads: 2
        )

        let layout = GemmaAttentionLayout(config)

        #expect(layout.headSize == 8)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
    }

    @Test("constructs Gemma model with greedy fast path and LoRA layers")
    func constructsGemmaModelWithGreedyFastPathAndLoRALayers() {
        let config = Self.smallConfig()

        let model = GemmaModel(config)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.modelType == "gemma")
        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    private static func smallConfig() -> GemmaConfiguration {
        GemmaConfiguration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            headDimensions: 4,
            vocabularySize: 64,
            kvHeads: 2
        )
    }
}
