import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Mamba architecture")
struct MambaArchitectureTests {
    @Test("decodes Mamba configuration aliases and auto rank")
    func decodesConfigurationAliasesAndAutoRank() throws {
        let config = try JSONDecoder.json5().decode(
            MambaConfiguration.self,
            from: Data(Self.aliasConfigJSON().utf8)
        )

        #expect(config.modelType == "mamba")
        #expect(config.hiddenSize == 16)
        #expect(config.intermediateSize == 32)
        #expect(config.stateSize == 4)
        #expect(config.hiddenLayers == 2)
        #expect(config.convKernel == 3)
        #expect(config.timeStepRank == 1)
        #expect(!config.useBias)
        #expect(config.useConvBias)
        #expect(config.tieWordEmbeddings)
        #expect(config.layerNormEpsilon == 1e-5)
    }

    @Test("builds Mamba mixer layout")
    func buildsMixerLayout() {
        let layout = MambaMixerLayout(Self.smallConfig())

        #expect(layout.hiddenSize == 16)
        #expect(layout.intermediateSize == 32)
        #expect(layout.stateSize == 4)
        #expect(layout.convKernel == 3)
        #expect(layout.timeStepRank == 2)
        #expect(layout.inputProjectionSize == 64)
        #expect(layout.stateProjectionSize == 10)
    }

    @Test("constructs model with Mamba caches, adapters, and greedy fast path")
    func constructsModelWithMambaCachesAdaptersAndGreedyFastPath() {
        let model = MambaModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(cache.count == 2)
        #expect(cache.allSatisfy { $0 is MambaCache })
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["in_proj", "x_proj", "dt_proj", "out_proj"])
    }

    @Test("tiny model produces finite logits and advances cache offsets")
    func tinyModelProducesFiniteLogitsAndAdvancesCacheOffsets() {
        Device.withDefaultDevice(.cpu) {
            let model = MambaModel(Self.smallConfig())
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
        let model = MambaModel(Self.smallConfig())
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

    private static func smallConfig(hiddenLayers: Int = 1) -> MambaConfiguration {
        MambaConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            stateSize: 4,
            hiddenLayers: hiddenLayers,
            convKernel: 3,
            timeStepRank: 2
        )
    }

    private static func aliasConfigJSON() -> String {
        """
        {
            "model_type": "mamba",
            "vocab_size": 64,
            "d_model": 16,
            "d_inner": 32,
            "d_state": 4,
            "n_layer": 2,
            "d_conv": 3,
            "bias": false,
            "conv_bias": true,
            "time_step_rank": "auto"
        }
        """
    }
}
