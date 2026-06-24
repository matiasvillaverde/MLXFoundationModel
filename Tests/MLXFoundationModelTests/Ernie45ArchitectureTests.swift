import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("ERNIE 4.5 architecture")
struct Ernie45ArchitectureTests {
    @Test("decodes ERNIE 4.5 configuration with project defaults")
    func decodesConfigurationWithDefaults() throws {
        let json = #"""
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "vocab_size": 64
        }
        """#

        let config = try JSONDecoder.json5().decode(
            Ernie45Configuration.self,
            from: Data(json.utf8)
        )

        #expect(config.hiddenSize == 16)
        #expect(config.intermediateSize == 32)
        #expect(config.maxPositionEmbeddings == 131_072)
        #expect(config.numAttentionHeads == 4)
        #expect(config.numKeyValueHeads == 2)
        #expect(config.headDim == nil)
        #expect(config.numHiddenLayers == 1)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.vocabularySize == 64)
        #expect(config.ropeTheta == 500_000)
        #expect(config.useBias == false)
        #expect(config.tieWordEmbeddings == true)
    }

    @Test("builds explicit ERNIE attention layout")
    func buildsAttentionLayout() {
        let layout = Ernie45AttentionLayout(Self.smallConfig(headDim: 8))

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 8)
        #expect(layout.queryProjectionSize == 32)
        #expect(layout.keyValueProjectionSize == 16)
        #expect(layout.attentionScale == 1 / Float(8).squareRoot())
    }

    @Test("uses hidden/head fallback when head_dim is absent")
    func usesFallbackHeadDimension() {
        let layout = Ernie45AttentionLayout(Self.smallConfig(headDim: nil))

        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
    }

    @Test("constructs ERNIE model with greedy fast path and LoRA layers")
    func constructsModelWithGreedyFastPathAndLoRALayers() {
        let tiedModel = Ernie45Model(Self.smallConfig())
        let untiedModel = Ernie45Model(Self.smallConfig(tieWordEmbeddings: false))
        let loraTargets = tiedModel.loraLinearLayers()
        let _: any GreedyTokenModel = tiedModel
        let _: any GreedyTokenModel = untiedModel

        #expect(tiedModel.vocabularySize == 64)
        #expect(tiedModel.kvHeads == [2])
        #expect(untiedModel.kvHeads == [2])
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("runs a small ERNIE forward pass")
    func runsSmallForwardPass() {
        Device.withDefaultDevice(.cpu) {
            let model = Ernie45Model(Self.smallConfig())
            let tokens = MLXArray([Int32(1), Int32(2)]).reshaped([1, 2])

            let logits = model(tokens, cache: nil)

            #expect(logits.shape == [1, 2, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    private static func smallConfig(
        headDim: Int? = 4,
        tieWordEmbeddings: Bool = true
    ) -> Ernie45Configuration {
        Ernie45Configuration(
            hiddenSize: 16,
            intermediateSize: 32,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: headDim,
            numHiddenLayers: 1,
            vocabularySize: 64,
            ropeTheta: 10_000,
            tieWordEmbeddings: tieWordEmbeddings
        )
    }
}
