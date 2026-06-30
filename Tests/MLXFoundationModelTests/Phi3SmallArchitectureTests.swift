import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Phi-3-small architecture")
struct Phi3SmallArchitectureTests {
    @Test("decodes real Phi-3-small configuration")
    func decodesRealConfiguration() throws {
        let config = try JSONDecoder.json5().decode(
            Phi3SmallConfiguration.self,
            from: Data(Self.realConfigJSON.utf8)
        )

        #expect(config.modelType == "phi3small")
        #expect(config.hiddenSize == 4_096)
        #expect(config.intermediateSize == 14_336)
        #expect(config.hiddenLayers == 32)
        #expect(config.attentionHeads == 32)
        #expect(config.keyValueHeads == 8)
        #expect(config.vocabularySize == 100_352)
        #expect(config.denseAttentionEveryN == 2)
        #expect(config.mupEmbeddingMultiplier == 10)
        #expect(config.mupWidthMultiplier == 8)
    }

    @Test("builds attention and sparse-mask plans")
    func buildsAttentionAndSparseMaskPlans() throws {
        try Device.withDefaultDevice(.cpu) {
            let config = Self.smallConfig(mupAttentionMultiplier: 2)
            let layout = Phi3SmallAttentionLayout(config)
            let sparse = try #require(Phi3SmallBlockSparsePlan(config, layout: layout, layerIndex: 0))
            let dense = Phi3SmallBlockSparsePlan(config, layout: layout, layerIndex: 1)

            #expect(layout.packedProjectionSize == 32)
            #expect(layout.queryHeadsPerKeyValueHead == 2)
            #expect(layout.headDimensions == 4)
            #expect(layout.attentionScale == 0.5)
            #expect(dense == nil)

            let mask = sparse.denseMask(queryLength: 3, keyLength: 5)
            eval(mask)
            #expect(mask.shape == [2, 2, 3, 5])
        }
    }

    @Test("tiny Phi-3-small model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Phi3SmallModel(Self.smallConfig())
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 128])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("greedy token path uses tied embedding logits")
    func greedyTokenPathUsesTiedEmbeddingLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Phi3SmallModel(Self.smallConfig())
            let output = model.greedyToken(
                LMInput.Text(tokens: MLXArray([1, 2, 3])),
                cache: nil,
                state: nil
            )
            eval(output.token)

            #expect(output.token.shape == [1])
        }
    }

    @Test("dummy tokenizer ids are suppressed for real vocabulary")
    func dummyTokenizerIDsAreSuppressed() {
        Device.withDefaultDevice(.cpu) {
            let config = Self.smallConfig(vocabularySize: 100_352)
            let model = Phi3SmallModel(config)
            let logits = model(MLXArray([1]).reshaped(1, 1), cache: nil)
            eval(logits)

            let values = logits[0, 0, MLXArray([100_256, 100_258]).asType(.uint32)]
                .asArray(Float.self)
            #expect(values.allSatisfy { $0 == -Float.infinity })
        }
    }

    private static func smallConfig(
        vocabularySize: Int = 128,
        mupAttentionMultiplier: Float = 1
    ) -> Phi3SmallConfiguration {
        Phi3SmallConfiguration(
            hiddenSize: 16,
            denseAttentionEveryN: 2,
            intermediateSize: 8,
            hiddenLayers: 2,
            attentionHeads: 4,
            vocabularySize: vocabularySize,
            keyValueHeads: 2,
            mupAttentionMultiplier: mupAttentionMultiplier,
            blockSparseBlockSize: 32,
            blockSparseLocalBlocks: 2,
            blockSparseVerticalStride: 2
        )
    }

    private static var realConfigJSON: String {
        #"""
        {
            "model_type": "phi3small",
            "hidden_size": 4096,
            "dense_attention_every_n_layers": 2,
            "ff_intermediate_size": 14336,
            "gegelu_limit": 20.0,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "layer_norm_epsilon": 0.00001,
            "vocab_size": 100352,
            "num_key_value_heads": 8,
            "mup_attn_multiplier": 1.0,
            "mup_use_scaling": true,
            "mup_embedding_multiplier": 10.0,
            "mup_width_multiplier": 8.0,
            "rope_embedding_base": 1000000,
            "rope_position_scale": 1.0,
            "blocksparse_block_size": 64,
            "blocksparse_num_local_blocks": 16,
            "blocksparse_vert_stride": 8
        }
        """#
    }
}
