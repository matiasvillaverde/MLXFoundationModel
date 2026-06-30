import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("EXAONE architecture")
struct ExaoneArchitectureTests {
    @Test("decodes EXAONE configuration with RoPE scaling defaults")
    func decodesConfigurationWithRoPEScalingDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            ExaoneConfiguration.self,
            from: Data(Self.configJSON().utf8)
        )

        #expect(config.modelType == "exaone")
        #expect(config.hiddenLayers == 30)
        #expect(config.resolvedHeadDim == 80)
        #expect(config.kvHeads == 8)
        #expect(config.ropeTheta == 1_000_000)
        #expect(config.ropeScaling?["rope_type"] == .string("llama3"))
        #expect(config.ropeScaling?["factor"]?.asFloat() == 8)
        #expect(config.tieWordEmbeddings)
        #expect(!config.attentionBias)
        #expect(!config.mlpBias)
    }

    @Test("builds grouped-query attention layout")
    func buildsGroupedQueryAttentionLayout() {
        let layout = ExaoneAttentionLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headDim == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = ExaoneModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny model produces finite logits with cache")
    func tinyModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = ExaoneModel(Self.smallConfig())
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(prefill)).item(Bool.self))
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer strips tied language head tensors")
    func sanitizerStripsTiedLanguageHeadTensors() {
        let model = ExaoneModel(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "transformer.wte.weight": MLXArray.ones([64, 16])
        ])

        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
        #expect(sanitized["transformer.wte.weight"] != nil)
    }

    private static func smallConfig(hiddenLayers: Int = 1) -> ExaoneConfiguration {
        ExaoneConfiguration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            intermediateSize: 32,
            attentionHeads: 4,
            vocabularySize: 64,
            ropeTheta: 1_000_000,
            kvHeads: 2,
            headDim: 4,
            maxPositionEmbeddings: 64,
            ropeScaling: [
                "factor": .float(8),
                "rope_type": .string("llama3")
            ]
        )
    }

    private static func configJSON() -> String {
        """
        {
            "model_type": "exaone",
            "hidden_size": 2560,
            "num_layers": 30,
            "intermediate_size": 7168,
            "num_attention_heads": 32,
            "vocab_size": 102400,
            "rope_theta": 1000000,
            "layer_norm_epsilon": 0.00001,
            "num_key_value_heads": 8,
            "head_dim": 80,
            "max_position_embeddings": 32768,
            "rope_scaling": {
                "factor": 8.0,
                "high_freq_factor": 4.0,
                "low_freq_factor": 1.0,
                "original_max_position_embeddings": 8192,
                "rope_type": "llama3"
            },
            "tie_word_embeddings": true
        }
        """
    }
}
