import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Mamba2 architecture")
struct Mamba2ArchitectureTests {
    @Test("decodes Mamba2 configuration with checkpoint aliases")
    func decodesConfigurationWithCheckpointAliases() throws {
        let config = try JSONDecoder.json5().decode(
            Mamba2Configuration.self,
            from: Data(Self.checkpointConfigJSON().utf8)
        )

        #expect(config.modelType == "mamba2")
        #expect(config.hiddenSize == 16)
        #expect(config.numHeads == 4)
        #expect(config.headDim == 4)
        #expect(config.intermediateSize == 16)
        #expect(config.stateSize == 8)
        #expect(config.hiddenLayers == 2)
        #expect(config.convKernel == 3)
        #expect(config.groups == 2)
        #expect(config.timeStepRank == 1)
        #expect(!config.useBias)
        #expect(config.useConvBias)
        #expect(config.tieWordEmbeddings)
        #expect(config.timeStepLimitMin == 0)
        #expect(config.timeStepLimitMax.isInfinite)
    }

    @Test("builds Mamba2 mixer layout")
    func buildsMixerLayout() {
        let layout = Mamba2MixerLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.heads == 4)
        #expect(layout.headDim == 4)
        #expect(layout.stateSize == 4)
        #expect(layout.groups == 2)
        #expect(layout.convKernel == 3)
        #expect(layout.intermediateSize == 16)
        #expect(layout.convInputSize == 32)
        #expect(layout.inputProjectionSize == 52)
    }

    @Test("constructs model with Mamba caches, adapters, and greedy fast path")
    func constructsModelWithMambaCachesAdaptersAndGreedyFastPath() {
        let model = Mamba2Model(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(cache.count == 2)
        #expect(cache.allSatisfy { $0 is MambaCache })
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["in_proj", "out_proj"])
    }

    @Test("tiny model produces finite logits and advances cache offsets")
    func tinyModelProducesFiniteLogitsAndAdvancesCacheOffsets() {
        Device.withDefaultDevice(.cpu) {
            let model = Mamba2Model(Self.smallConfig())
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache.map(\.offset) == [3])
            #expect(all(isFinite(prefill)).item(Bool.self))
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer reshapes convolution weights and strips tied language head")
    func sanitizerReshapesConvolutionWeightsAndStripsTiedLanguageHead() {
        let model = Mamba2Model(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "backbone.layers.0.mixer.conv1d.weight": MLXArray.ones([32, 1, 3]),
            "lm_head.weight": MLXArray.ones([64, 16]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1])
        ])

        #expect(sanitized["backbone.layers.0.mixer.conv1d.weight"]?.shape == [32, 3, 1])
        #expect(sanitized["lm_head.weight"] == nil)
        #expect(sanitized["lm_head.scales"] == nil)
        #expect(sanitized["lm_head.biases"] == nil)
    }

    private static func smallConfig(hiddenLayers: Int = 1) -> Mamba2Configuration {
        Mamba2Configuration(
            numHeads: 4,
            headDim: 4,
            vocabularySize: 64,
            hiddenSize: 16,
            stateSize: 4,
            hiddenLayers: hiddenLayers,
            convKernel: 3,
            groups: 2,
            timeStepRank: 2
        )
    }

    private static func checkpointConfigJSON() -> String {
        """
        {
            "model_type": "mamba2",
            "num_heads": 4,
            "head_dim": 4,
            "vocab_size": 64,
            "hidden_size": 16,
            "state_size": 8,
            "num_hidden_layers": 2,
            "layer_norm_epsilon": 1e-5,
            "conv_kernel": 3,
            "n_groups": 2,
            "use_bias": false,
            "use_conv_bias": true,
            "tie_word_embeddings": true,
            "time_step_limit": [0.0, Infinity],
            "time_step_rank": "auto"
        }
        """
    }
}
