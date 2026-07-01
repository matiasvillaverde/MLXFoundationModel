import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Plamo2 architecture")
struct Plamo2ArchitectureTests {
    @Test("decodes Plamo2 1B configuration")
    func decodesConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            Plamo2Configuration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "plamo2")
        #expect(config.hiddenSize == 2_048)
        #expect(config.hiddenLayers == 16)
        #expect(config.attentionHeads == 16)
        #expect(config.keyValueHeads == 1)
        #expect(config.headSize == 128)
        #expect(config.mambaHeads == 32)
        #expect(config.mambaStateSize == 64)
        #expect(config.mambaStep == 2)
        #expect(config.intermediateSize == 8_192)
        #expect(config.vocabularySize == 100_000)
    }

    @Test("builds layer, attention, and Mamba plans")
    func buildsPlans() {
        let config = Self.smallConfig(hiddenLayers: 4)
        let plan = Plamo2LayerPlan(config)
        let attention = Plamo2AttentionLayout(config)
        let mamba = Plamo2MambaLayout(config)

        #expect(plan.kinds == [.mamba, .attention, .mamba, .attention])
        #expect(plan.firstMambaIndex == 0)
        #expect(plan.firstAttentionIndex == 1)
        #expect(plan.kvHeads == [0, 1, 0, 1])
        #expect(attention.projectionDimensions == 24)
        #expect(attention.scale == 0.5)
        #expect(mamba.intermediateSize == 16)
        #expect(mamba.timeStepDimensions == 64)
        #expect(mamba.stateProjectionDimensions == 72)
    }

    @Test("constructs model with mixed caches, adapters, and greedy fast path")
    func constructsModelWithMixedCachesAdaptersAndGreedyFastPath() {
        let model = Plamo2Model(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [0, 1])
        #expect(cache.count == 2)
        #expect(cache[0] is MambaCache)
        #expect(cache[1] is KVCacheSimple)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["in_proj", "bcdt_proj", "dt_proj", "out_proj"])
        #expect(loraTargets[1].1 == ["qkv_proj", "o_proj"])
    }

    @Test("tiny tied model produces finite logits with Mamba and attention layers")
    func tinyTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Plamo2Model(Self.smallConfig(hiddenLayers: 2))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = Plamo2Model(
                Self.smallConfig(hiddenLayers: 2, tieWordEmbeddings: false)
            )
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(cache[1].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer moves convolution weights and strips tied output head")
    func sanitizerMovesConvolutionWeightsAndStripsTiedHead() {
        Device.withDefaultDevice(.cpu) {
            let model = Plamo2Model(Self.smallConfig(hiddenLayers: 1))
            let weights = [
                "lm_head.weight": MLXArray.ones([2, 2]),
                "lm_head.scales": MLXArray.ones([1]),
                "lm_head.biases": MLXArray.ones([1]),
                "model.layers.layers.0.mixer.conv1d.weight": MLXArray.zeros([4, 1, 3])
            ]
            let sanitized = model.sanitize(weights: weights)
            let conv = sanitized["model.layers.layers.0.mixer.conv1d.weight"]

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(conv?.shape == [4, 3, 1])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 2,
        tieWordEmbeddings: Bool = true
    ) -> Plamo2Configuration {
        Plamo2Configuration(
            hiddenSize: 16,
            hiddenLayers: hiddenLayers,
            tieWordEmbeddings: tieWordEmbeddings,
            attentionHeads: 4,
            keyValueHeads: 1,
            headSize: 4,
            maxPositionEmbeddings: 64,
            attentionWindowSize: 64,
            mambaStateSize: 4,
            mambaConvKernel: 3,
            mambaHeads: 4,
            mambaStep: 2,
            mambaChunkSize: 16,
            intermediateSize: 32,
            vocabularySize: 64
        )
    }

    private static let configJSON = """
        {
            "model_type": "plamo2",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "rms_norm_eps": 1e-06,
            "tie_word_embeddings": true,
            "num_attention_heads": 16,
            "num_key_value_heads": 1,
            "hidden_size_per_head": 128,
            "max_position_embeddings": 10485760,
            "attention_window_size": 2048,
            "mamba_d_state": 64,
            "mamba_d_conv": 4,
            "mamba_num_heads": 32,
            "mamba_step": 2,
            "mamba_chunk_size": 256,
            "mamba_enabled": true,
            "intermediate_size": 8192,
            "vocab_size": 100000
        }
        """
}
