import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Mistral3 text architecture")
struct Mistral3TextArchitectureTests {
    @Test("decodes nested Mistral3 text configuration with project defaults")
    func decodesNestedMistral3TextConfigurationWithDefaults() throws {
        let json = #"""
        {
            "tie_word_embeddings": true,
            "text_config": {
                "hidden_size": 16,
                "num_hidden_layers": 2,
                "intermediate_size": 32,
                "num_attention_heads": 4,
                "rms_norm_eps": 0.00001,
                "vocab_size": 64,
                "max_position_embeddings": 128
            }
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Mistral3TextConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "ministral3")
        #expect(config.kvHeads == 4)
        #expect(config.ropeTheta == 10_000)
        #expect(config.tieWordEmbeddings == true)
        #expect(config.layerTypes == ["full_attention", "full_attention"])
    }

    @Test("builds Mistral3 attention layout, layer schedule, and scaling plan")
    func buildsMistral3LayoutScheduleAndScalingPlan() throws {
        let config = Self.smallConfig(
            hiddenLayers: 3,
            maxPositionEmbeddings: 128,
            ropeParameters: [
                "llama_4_scaling_beta": .float(0.25),
                "original_max_position_embeddings": .int(4)
            ],
            layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
            slidingWindow: 16
        )

        let layout = Mistral3AttentionLayout(config)
        let schedule = Mistral3LayerSchedule(config)
        let scalePlan = Mistral3AttentionScalePlan(config)
        let scale = scalePlan.values(start: 3, count: 6, dtype: .float32)
        eval(scale)

        Self.expectSmallLayout(layout)
        Self.expectMixedSchedule(schedule)
        #expect(scalePlan.usesPositionScaling)

        let values = scale.asArray(Float.self)
        #expect(values[0] == 1)
        #expect(values[1] > 1)
        #expect(values[2] == values[1])
        #expect(values[5] > values[1])
    }

    @Test("constructs Mistral3 model with mixed caches, adapters, and greedy fast path")
    func constructsMistral3ModelWithMixedCachesAdaptersAndGreedyFastPath() throws {
        let model = Mistral3TextModel(
            Self.smallConfig(
                hiddenLayers: 3,
                layerTypes: ["sliding_attention", "full_attention", "sliding_attention"],
                slidingWindow: 16
            )
        )
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2, 2])
        _ = try #require(cache[0] as? RotatingKVCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        _ = try #require(cache[2] as? RotatingKVCache)
        #expect(loraTargets.count == 3)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny Mistral3 model produces finite logits")
    func tinyMistral3ModelProducesFiniteLogits() {
        let model = Mistral3TextModel(Self.smallConfig(hiddenLayers: 1))
        let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(logits)

        #expect(logits.shape == [1, 3, 64])
        #expect(all(isFinite(logits)).item(Bool.self))
    }

    @Test("sanitizer unwraps VLM language model weights and drops unused tensors")
    func sanitizerUnwrapsLanguageModelWeightsAndDropsUnusedTensors() {
        let model = Mistral3TextModel(Self.smallConfig(tieWordEmbeddings: true))
        let sanitized = model.sanitize(weights: [
            "language_model.lm_head.weight": MLXArray.ones([64, 16]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "language_model.model.embed_tokens.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
        #expect(sanitized["model.embed_tokens.weight"] != nil)
        #expect(sanitized["language_model.model.embed_tokens.weight"] == nil)
    }

    private static func smallConfig(
        hiddenLayers: Int = 1,
        maxPositionEmbeddings: Int? = 64,
        ropeParameters: [String: StringOrNumber] = [:],
        tieWordEmbeddings: Bool = true,
        layerTypes: [String] = [],
        slidingWindow: Int? = nil
    ) -> Mistral3TextConfiguration {
        Mistral3TextConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            maxPositionEmbeddings: maxPositionEmbeddings,
            kvHeads: 2,
            ropeParameters: ropeParameters.isEmpty ? nil : ropeParameters,
            tieWordEmbeddings: tieWordEmbeddings,
            layerTypes: layerTypes.isEmpty ? nil : layerTypes,
            slidingWindow: slidingWindow
        )
    }

    private static func expectSmallLayout(_ layout: Mistral3AttentionLayout) {
        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    private static func expectMixedSchedule(_ schedule: Mistral3LayerSchedule) {
        #expect(schedule.fullAttentionMaskLayerIndex == 1)
        #expect(schedule.slidingAttentionMaskLayerIndex == 0)
        #expect(schedule.usesSlidingWindow(at: 0))
        #expect(!schedule.usesSlidingWindow(at: 1))
    }
}
