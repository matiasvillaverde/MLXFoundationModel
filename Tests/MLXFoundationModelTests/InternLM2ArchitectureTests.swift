import Foundation
@testable import MLXLocalModels
import Testing

@Suite("InternLM2 architecture")
struct InternLM2ArchitectureTests {
    @Test("decodes InternLM2 configuration with defaults")
    func decodesInternLM2ConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "num_key_value_heads": 2,
            "rope_scaling": {
                "type": "dynamic",
                "factor": 2.0
            }
        }
        """#

        let config = try JSONDecoder.json5().decode(
            InternLM2Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.attentionHeads == 4)
        #expect(config.kvHeads == 2)
        #expect(config.maxPositionEmbeddings == 32_768)
        #expect(config.ropeTheta == 10_000)
        #expect(config.tieWordEmbeddings == false)
        #expect(config.bias == true)
    }

    @Test("builds packed InternLM2 attention layout")
    func buildsPackedInternLM2AttentionLayout() {
        let config = Self.smallConfig()

        let layout = InternLM2AttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.keyValueGroups == 2)
        #expect(layout.headSize == 4)
        #expect(layout.packedProjectionSize == 32)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("plans dynamic RoPE scaling from sequence length")
    func plansDynamicRoPEScalingFromSequenceLength() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("dynamic"), "factor": .float(2)]
        )

        let plan = InternLM2RoPEPlan(config, dimensions: 4)

        #expect(plan.positionScale == 1)
        #expect(plan.dynamicFactor == 2)
        #expect(plan.adjustedBase(sequenceLength: 16) == 10_000)
        #expect(plan.adjustedBase(sequenceLength: 64) > 10_000)
    }

    @Test("plans linear RoPE as position scaling only")
    func plansLinearRoPEAsPositionScalingOnly() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )

        let plan = InternLM2RoPEPlan(config, dimensions: 4)

        #expect(plan.positionScale == 0.25)
        #expect(plan.dynamicFactor == nil)
        #expect(plan.adjustedBase(sequenceLength: 64) == 10_000)
    }

    @Test("constructs InternLM2 model with greedy fast path and packed LoRA target")
    func constructsInternLM2ModelWithGreedyFastPathAndPackedLoRATarget() {
        let config = Self.smallConfig()

        let model = InternLM2Model(config)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["wqkv"])
    }

    private static func smallConfig(
        ropeScaling: [String: StringOrNumber] = ["type": .string("dynamic"), "factor": .float(2)]
    ) -> InternLM2Configuration {
        InternLM2Configuration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: 32,
            ropeScaling: ropeScaling
        )
    }
}
