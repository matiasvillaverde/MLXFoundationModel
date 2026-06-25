import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Granite MoE Hybrid architecture")
struct GraniteMoeHybridArchitectureTests {
    @Test("decodes typed layer schedule")
    func decodesTypedLayerSchedule() throws {
        let json = #"""
        {
            "num_hidden_layers": 2,
            "layer_types": ["mamba", "attention"]
        }
        """#

        let config = try JSONDecoder.json5().decode(
            GraniteMoeHybridConfiguration.self,
            from: Data(json.utf8)
        )

        #expect(config.modelType == "granitemoehybrid")
        #expect(config.layerTypes == [.mamba, .attention])
        #expect(GraniteMoeHybridLayerPlan(config).firstMambaIndex == 0)
        #expect(GraniteMoeHybridLayerPlan(config).firstAttentionIndex == 1)
    }

    @Test("rejects invalid layer schedule")
    func rejectsInvalidLayerSchedule() {
        let json = #"""
        {
            "num_hidden_layers": 2,
            "layer_types": ["mamba"]
        }
        """#

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                GraniteMoeHybridConfiguration.self,
                from: Data(json.utf8)
            )
        }
    }

    @Test("computes attention, mamba, and MoE layouts")
    func computesLayouts() {
        let config = Self.smallConfig(
            layerTypes: [.mamba, .attention],
            usesMoE: true
        )
        let attention = GraniteMoeHybridAttentionLayout(config)
        let mamba = GraniteMoeHybridMambaLayout(config)
        let moe = GraniteMoeHybridMoEPlan(config)

        #expect(attention.queryDimensions == 16)
        #expect(attention.keyValueDimensions == 8)
        #expect(attention.usesRotaryPosition)
        #expect(mamba.intermediateSize == 8)
        #expect(mamba.convolutionDimensions == 16)
        #expect(mamba.projectionDimensions == 26)
        #expect(moe.expertCount == 2)
        #expect(moe.selectedExpertCount == 1)
    }

    @Test("cache types and LoRA targets follow layer schedule")
    func cacheTypesAndLoRATargetsFollowLayerSchedule() throws {
        let model = GraniteMoeHybridModel(
            Self.smallConfig(layerTypes: [.mamba, .attention])
        )
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.kvHeads == [0, 2])
        _ = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(loraTargets.count == 1)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny Granite MoE Hybrid model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = GraniteMoeHybridModel(
                Self.smallConfig(layerTypes: [.mamba, .attention])
            )
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: [])

            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer remaps later-layer block-sparse MoE weights")
    func sanitizerRemapsLaterLayerBlockSparseMoEWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GraniteMoeHybridModel(
                Self.smallConfig(
                    layerTypes: [.attention, .mamba],
                    usesMoE: true,
                    tieWordEmbeddings: true
                )
            )
            let sanitized = model.sanitize(weights: Self.blockSparseWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["model.layers.1.block_sparse_moe.input_linear.weight"] == nil)

            let conv = try #require(sanitized["model.layers.1.mamba.conv1d.weight"])
            let gate = try #require(
                sanitized["model.layers.1.block_sparse_moe.switch_mlp.gate_proj.weight"]
            )
            let upProjection = try #require(
                sanitized["model.layers.1.block_sparse_moe.switch_mlp.up_proj.weight"]
            )
            let down = try #require(
                sanitized["model.layers.1.block_sparse_moe.switch_mlp.down_proj.weight"]
            )

            eval(conv, gate, upProjection, down)
            #expect(conv.shape == [2, 4, 3])
            #expect(gate.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(down.shape == [2, 2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [2, 2, 2, 2])
        }
    }

    @Test("sanitizer remaps dense shared MLP weights")
    func sanitizerRemapsDenseSharedMLPWeights() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = GraniteMoeHybridModel(
                Self.smallConfig(layerTypes: [.attention, .mamba])
            )
            let sanitized = model.sanitize(weights: Self.denseSharedWeights())

            #expect(sanitized["model.layers.1.shared_mlp.input_linear.weight"] == nil)

            let gate = try #require(sanitized["model.layers.1.mlp.gate_proj.weight"])
            let upProjection = try #require(sanitized["model.layers.1.mlp.up_proj.weight"])
            let down = try #require(sanitized["model.layers.1.mlp.down_proj.weight"])

            eval(gate, upProjection, down)
            #expect(gate.shape == [2, 2])
            #expect(upProjection.shape == [2, 2])
            #expect(down.shape == [2, 2])
            #expect(gate.asArray(Float.self) == [1, 1, 1, 1])
            #expect(upProjection.asArray(Float.self) == [2, 2, 2, 2])
        }
    }

    private static func smallConfig(
        layerTypes: [GraniteMoeHybridLayerKind],
        usesMoE: Bool = false,
        tieWordEmbeddings: Bool = false
    ) -> GraniteMoeHybridConfiguration {
        GraniteMoeHybridConfiguration(
            vocabularySize: 64,
            hiddenSize: 16,
            intermediateSize: 32,
            hiddenLayers: layerTypes.count,
            maxPositionEmbeddings: 128,
            attentionHeads: 4,
            kvHeads: 2,
            attentionBias: false,
            embeddingMultiplier: 1,
            attentionMultiplier: 1,
            logitsScaling: 1,
            residualMultiplier: 1,
            layerTypes: layerTypes,
            rmsNormEps: 1e-6,
            ropeTheta: 10_000,
            numLocalExperts: usesMoE ? 2 : nil,
            numExpertsPerToken: usesMoE ? 1 : nil,
            sharedIntermediateSize: usesMoE ? 8 : nil,
            mambaHeads: 2,
            mambaHeadDim: 4,
            mambaProjBias: false,
            mambaStateDim: 4,
            mambaConvKernel: 2,
            mambaGroups: 1,
            mambaConvBias: false,
            mlpBias: false,
            positionEmbeddingType: "rope",
            tieWordEmbeddings: tieWordEmbeddings
        )
    }

    private static func blockSparseWeights() -> [String: MLXArray] {
        [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "model.layers.1.mamba.conv1d.weight": MLXArray.ones([2, 3, 4]),
            "model.layers.1.block_sparse_moe.input_linear.weight": concatenated(
                [
                    MLXArray.ones([2, 2, 2]),
                    Self.filledArray(shape: [2, 2, 2], value: 2)
                ],
                axis: 1
            ),
            "model.layers.1.block_sparse_moe.output_linear.weight": Self.filledArray(
                shape: [2, 2, 2],
                value: 3
            )
        ]
    }

    private static func denseSharedWeights() -> [String: MLXArray] {
        [
            "model.layers.1.shared_mlp.input_linear.weight": concatenated(
                [
                    MLXArray.ones([2, 2]),
                    Self.filledArray(shape: [2, 2], value: 2)
                ],
                axis: 0
            ),
            "model.layers.1.shared_mlp.output_linear.weight": Self.filledArray(
                shape: [2, 2],
                value: 3
            )
        ]
    }

    private static func filledArray(shape: [Int], value: Float) -> MLXArray {
        MLXArray([Float](repeating: value, count: shape.reduce(1, *))).reshaped(shape)
    }
}
