import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Llama architecture")
struct LlamaArchitectureTests {
    @Test("decodes Llama configuration with project defaults")
    func decodesLlamaConfigurationWithDefaults() throws {
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
            LlamaConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.hiddenLayers == 1)
        #expect(config.intermediateSize == 32)
        #expect(config.attentionHeads == 4)
        #expect(config.headDimensions == nil)
        #expect(config.resolvedHeadDimensions == 4)
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.attentionBias == false)
        #expect(config.mlpBias == false)
    }

    @Test("builds explicit Llama attention layout")
    func buildsExplicitLlamaAttentionLayout() {
        let config = Self.smallConfig()

        let layout = LlamaAttentionLayout(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("uses provided Llama head dimensions")
    func usesProvidedLlamaHeadDimensions() {
        let config = LlamaConfiguration(
            hiddenSize: 24,
            hiddenLayers: 1,
            intermediateSize: 48,
            attentionHeads: 4,
            headDimensions: 8,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2
        )

        let layout = LlamaAttentionLayout(config)

        #expect(layout.headSize == 8)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
    }

    @Test("plans linear Llama RoPE scaling")
    func plansLinearLlamaRoPEScaling() {
        let config = Self.smallConfig(
            ropeScaling: ["type": .string("linear"), "factor": .float(4)]
        )

        let plan = LlamaRoPEPlan(config, dimensions: 4)

        #expect(plan.kind == .linear(factor: 4))
        #expect(plan.positionScale == 0.25)
        #expect(plan.adjustedBase(sequenceLength: 128) == 10_000)
    }

    @Test("plans dynamic Llama RoPE base adjustment")
    func plansDynamicLlamaRoPEBaseAdjustment() {
        let config = Self.smallConfig(
            maxPositionEmbeddings: 32,
            ropeScaling: ["rope_type": .string("dynamic"), "factor": .float(2)]
        )

        let plan = LlamaRoPEPlan(config, dimensions: 4)

        #expect(plan.kind == .dynamic(factor: 2))
        #expect(plan.positionScale == 1)
        #expect(plan.adjustedBase(sequenceLength: 16) == 10_000)
        #expect(plan.adjustedBase(sequenceLength: 64) > 10_000)
    }

    @Test("plans Llama 3 RoPE scaling")
    func plansLlama3RoPEScaling() {
        let config = Self.smallConfig(
            ropeScaling: [
                "rope_type": .string("llama3"),
                "factor": .float(8),
                "low_freq_factor": .float(1),
                "high_freq_factor": .float(4),
                "original_max_position_embeddings": .int(8_192)
            ]
        )

        let plan = LlamaRoPEPlan(config, dimensions: 4)

        #expect(
            plan.kind == .llama3(
                factor: 8,
                lowFrequencyFactor: 1,
                highFrequencyFactor: 4,
                originalMaxPositionEmbeddings: 8_192
            )
        )
        #expect(plan.positionScale == 1)
    }

    @Test("constructs Llama model with greedy fast path and LoRA layers")
    func constructsLlamaModelWithGreedyFastPathAndLoRALayers() {
        let model = LlamaModel(Self.smallConfig())
        let untiedModel = LlamaModel(Self.smallConfig(tieWordEmbeddings: false))
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model
        let _: any GreedyTokenModel = untiedModel

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2])
        #expect(untiedModel.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    private static func smallConfig(
        maxPositionEmbeddings: Int? = 64,
        ropeScaling: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = true
    ) -> LlamaConfiguration {
        LlamaConfiguration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            kvHeads: 2,
            maxPositionEmbeddings: maxPositionEmbeddings,
            ropeScaling: ropeScaling.isEmpty ? nil : ropeScaling,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
