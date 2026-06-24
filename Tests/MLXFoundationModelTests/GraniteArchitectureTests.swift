import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Granite architecture")
struct GraniteArchitectureTests {
    @Test("decodes Granite configuration with project defaults")
    func decodesGraniteConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            GraniteConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 4)
        #expect(config.logitsScaling == 8)
        #expect(config.attentionMultiplier == 1.0 / 64.0)
        #expect(config.embeddingMultiplier == 12)
        #expect(config.residualMultiplier == 0.22)
        #expect(config.ropeTheta == 10_000_000)
        #expect(config.tieWordEmbeddings == true)
    }

    @Test("builds explicit Granite attention layout")
    func buildsExplicitGraniteAttentionLayout() {
        let layout = GraniteAttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.25)
    }

    @Test("plans linear Granite RoPE scaling")
    func plansLinearGraniteRoPEScaling() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )

        let plan = GraniteRoPEPlan(config, dimensions: 4)

        #expect(plan.dimensions == 4)
        #expect(plan.base == 10_000_000)
        #expect(plan.scale == 0.25)
    }

    @Test("constructs Granite model with greedy fast path and LoRA layers")
    func constructsGraniteModelWithGreedyFastPathAndLoRALayers() {
        let tiedModel = GraniteModel(Self.smallConfig())
        let untiedModel = GraniteModel(Self.smallConfig(tieWordEmbeddings: false))
        let loraTargets = tiedModel.loraLinearLayers()
        let _: any GreedyTokenModel = tiedModel
        let _: any GreedyTokenModel = untiedModel

        #expect(tiedModel.vocabularySize == 64)
        #expect(tiedModel.kvHeads == [2])
        #expect(untiedModel.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("runs a small Granite forward pass")
    func runsSmallGraniteForwardPass() {
        Device.withDefaultDevice(.cpu) {
            let model = GraniteModel(Self.smallConfig())
            let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])

            let logits = model(tokens, cache: nil)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    private static func smallConfig(
        ropeScaling: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = true
    ) -> GraniteConfiguration {
        GraniteConfiguration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            logitsScaling: 2,
            attentionMultiplier: 0.25,
            embeddingMultiplier: 1,
            residualMultiplier: 0.5,
            kvHeads: 2,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
