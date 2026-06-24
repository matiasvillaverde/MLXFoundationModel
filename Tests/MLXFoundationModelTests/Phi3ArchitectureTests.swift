import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Phi3 architecture")
struct Phi3ArchitectureTests {
    @Test("decodes Phi3 configuration with project defaults")
    func decodesPhi3ConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Phi3Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 4)
        #expect(config.maxPositionEmbeddings == 4_096)
        #expect(config.originalMaxPositionEmbeddings == 4_096)
        #expect(config.partialRotaryFactor == 1)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("builds packed Phi3 attention layout")
    func buildsPackedPhi3AttentionLayout() {
        let config = Self.smallConfig(partialRotaryFactor: 0.5)

        let layout = Phi3AttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.rotaryDimensions == 2)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.packedProjectionSize == 32)
        #expect(layout.keySplitIndex == 16)
        #expect(layout.valueSplitIndex == 24)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("plans linear Phi3 RoPE scaling")
    func plansLinearPhi3RoPEScaling() {
        let config = Self.smallConfig(
            ropeScaling: Phi3RoPEScaling(
                longFactor: nil,
                shortFactor: nil,
                factor: 4,
                type: "linear",
                longMScale: nil,
                shortMScale: nil
            )
        )
        let layout = Phi3AttentionLayout(config)

        let plan = Phi3RotaryPlan(config, layout: layout)

        #expect(plan.dimensions == 4)
        #expect(plan.kind == .rope(scale: 0.25))
    }

    @Test("plans Phi3 LongRoPE from long factors")
    func plansPhi3LongRoPEFromLongFactors() {
        let config = Self.smallConfig(
            ropeScaling: Phi3RoPEScaling(
                longFactor: [1, 2, 3],
                shortFactor: [1, 1, 1],
                factor: nil,
                type: "longrope",
                longMScale: nil,
                shortMScale: nil
            ),
            maxPositionEmbeddings: 128,
            originalMaxPositionEmbeddings: 64
        )
        let layout = Phi3AttentionLayout(config)

        let plan = Phi3RotaryPlan(config, layout: layout)

        #expect(plan.maxPositionEmbeddings == 128)
        #expect(plan.originalMaxPositionEmbeddings == 64)
        #expect(plan.kind == .longRoPE(longFactor: [1, 2, 3]))
    }

    @Test("constructs Phi3 model with greedy fast path and LoRA layers")
    func constructsPhi3ModelWithGreedyFastPathAndLoRALayers() {
        let model = Phi3Model(Self.smallConfig())
        let tiedModel = Phi3Model(Self.smallConfig(tieWordEmbeddings: true))
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model
        let _: any GreedyTokenModel = tiedModel

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2])
        #expect(tiedModel.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["qkv_proj"])
    }

    private static func smallConfig(
        ropeScaling: Phi3RoPEScaling? = nil,
        partialRotaryFactor: Float = 1,
        maxPositionEmbeddings: Int = 64,
        originalMaxPositionEmbeddings: Int? = 64,
        tieWordEmbeddings: Bool = false
    ) -> Phi3Configuration {
        Phi3Configuration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            kvHeads: 2,
            ropeScaling: ropeScaling,
            partialRotaryFactor: partialRotaryFactor,
            maxPositionEmbeddings: maxPositionEmbeddings,
            originalMaxPositionEmbeddings: originalMaxPositionEmbeddings,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
