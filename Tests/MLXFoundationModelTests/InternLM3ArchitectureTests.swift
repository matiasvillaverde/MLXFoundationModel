import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("InternLM3 architecture")
struct InternLM3ArchitectureTests {
    @Test("decodes real InternLM3 configuration")
    func decodesRealConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            InternLM3Configuration.self,
            from: Data(Self.realConfigJSON.utf8)
        )

        #expect(config.modelType == "internlm3")
        #expect(config.hiddenSize == 4_096)
        #expect(config.hiddenLayers == 48)
        #expect(config.intermediateSize == 10_240)
        #expect(config.attentionHeads == 32)
        #expect(config.keyValueHeads == 2)
        #expect(config.resolvedHeadDim == 128)
        #expect(config.vocabularySize == 128_512)
        #expect(config.ropeTheta == 50_000_000)
        #expect(config.tieWordEmbeddings == false)
    }

    @Test("builds attention layout")
    func buildsAttentionLayout() {
        let layout = InternLM3AttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.keyValueGroups == 2)
        #expect(layout.headDim == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("plans InternLM3 dynamic RoPE scaling")
    func plansInternLM3DynamicRoPEScaling() {
        let plan = InternLM3RoPEPlan(Self.smallConfig(), dimensions: 4)

        #expect(plan.positionScale == 2)
        #expect(plan.dynamicBaseScale == 2)
        #expect(plan.adjustedBase(sequenceLength: 16) == 10_000)
        #expect(plan.adjustedBase(sequenceLength: 64) > 10_000)
    }

    @Test("plans linear RoPE scaling")
    func plansLinearRoPEScaling() {
        let config = Self.smallConfig(
            ropeScaling: ["rope_type": .string("linear"), "factor": .float(4)]
        )

        let plan = InternLM3RoPEPlan(config, dimensions: 4)

        #expect(plan.positionScale == 0.25)
        #expect(plan.dynamicBaseScale == nil)
        #expect(plan.adjustedBase(sequenceLength: 64) == 10_000)
    }

    @Test("tiny InternLM3 model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = InternLM3Model(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("constructs greedy fast path and LoRA targets")
    func constructsGreedyFastPathAndLoRATargets() {
        Device.withDefaultDevice(.cpu) {
            let model = InternLM3Model(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            let loraTargets = model.loraLinearLayers()
            eval(output.token)

            #expect(model.vocabularySize == 64)
            #expect(model.kvHeads == [2])
            #expect(output.token.shape == [1])
            #expect(loraTargets.count == 1)
            #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
        }
    }

    private static func smallConfig(
        ropeScaling: [String: StringOrNumber] = ["rope_type": .string("dynamic"), "factor": .float(6)]
    ) -> InternLM3Configuration {
        InternLM3Configuration(
            hiddenSize: 16,
            hiddenLayers: 1,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            maxPositionEmbeddings: 32,
            keyValueHeads: 2,
            ropeScaling: ropeScaling,
            headDim: 4
        )
    }

    private static var realConfigJSON: String {
        #"""
        {
            "architectures": [
                "InternLM3ForCausalLM"
            ],
            "bias": false,
            "head_dim": 128,
            "hidden_size": 4096,
            "intermediate_size": 10240,
            "max_position_embeddings": 32768,
            "model_type": "internlm3",
            "num_attention_heads": 32,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "qkv_bias": false,
            "rms_norm_eps": 0.00001,
            "rope_scaling": {
                "factor": 6.0,
                "rope_type": "dynamic"
            },
            "rope_theta": 50000000,
            "tie_word_embeddings": false,
            "vocab_size": 128512
        }
        """#
    }
}
