import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("Nemotron H architecture")
struct NemotronHArchitectureTests {
    @Test("decodes block pattern and time-step limits")
    func decodesBlockPatternAndTimeStepLimits() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self,
            from: Data(Self.configJSON(patternJSON: #"["M", "*", "E", "-"]"#).utf8)
        )

        #expect(config.blockPattern == [.mamba, .attention, .routedFeedForward, .feedForward])
        #expect(config.timeStepLimitMin == 0.001)
        #expect(config.timeStepLimitMax == 42.0)

        let plan = NemotronHLayerPlan(config)
        #expect(plan.firstMambaCacheIndex == 0)
        #expect(plan.firstAttentionCacheIndex == 1)
        #expect(plan.kvHeads == [0, 2])
    }

    @Test("decodes dense-only Nano-style config without MoE fields")
    func decodesDenseOnlyNanoStyleConfigWithoutMoEFields() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronHConfiguration.self,
            from: Data(Self.denseOnlyConfigJSON.utf8)
        )

        #expect(config.blockPattern == [.mamba, .attention, .feedForward])
        #expect(config.moeIntermediateSize == config.intermediateSize)
        #expect(config.moeSharedExpertIntermediateSize == config.intermediateSize)
        #expect(config.nRoutedExperts == 1)
        #expect(config.numExpertsPerTok == 1)

        let layout = NemotronHAttentionLayout(config)
        #expect(layout.queryDimensions == 40 * 128)
        #expect(layout.keyValueDimensions == 8 * 128)
    }

    @Test("rejects invalid block pattern")
    func rejectsInvalidBlockPattern() {
        let json = Self.configJSON(pattern: "M*")

        #expect(throws: DecodingError.self) {
            _ = try JSONDecoder.json5().decode(
                NemotronHConfiguration.self,
                from: Data(json.utf8)
            )
        }
    }

    @Test("computes attention, Mamba, and MoE layouts")
    func computesLayouts() {
        let config = Self.smallConfig(pattern: "M*E-")
        let attention = NemotronHAttentionLayout(config)
        let mamba = NemotronHMambaLayout(config)
        let moe = NemotronHMoEPlan(config)

        #expect(attention.queryDimensions == 16)
        #expect(attention.keyValueDimensions == 8)
        #expect(attention.scale == 0.5)
        #expect(mamba.intermediateSize == 8)
        #expect(mamba.convolutionDimensions == 12)
        #expect(mamba.projectionDimensions == 22)
        #expect(mamba.gatedNormGroupSize == 8)
        #expect(moe.routedExperts == 2)
        #expect(moe.expertsPerToken == 1)
    }

    @Test("cache types, KV heads, LoRA targets, and greedy path follow pattern")
    func cacheTypesKVHeadsLoRAAndGreedyPathFollowPattern() throws {
        let model = NemotronHModel(Self.smallConfig(pattern: "M*E-"))
        let cache = model.newCache(parameters: nil)
        let targets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.kvHeads == [0, 2])
        _ = try #require(cache[0] as? MambaCache)
        _ = try #require(cache[1] as? KVCacheSimple)
        #expect(targets.count == 1)
        #expect(targets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("routing keeps correction bias selection-only")
    func routingKeepsCorrectionBiasSelectionOnly() {
        let config = Self.smallConfig(
            pattern: "E",
            nRoutedExperts: 4,
            nGroup: 2,
            topkGroup: 1,
            routedScalingFactor: 2
        )
        let plan = NemotronHMoEPlan(config)
        let logits = MLXArray([Float(4), Float(3), Float(1), Float(10)]).reshaped(1, 1, 4)
        let bias = MLXArray([Float(0), Float(0), Float(10), Float(0)])

        let route = plan.route(logits: logits, correctionBias: bias, outputDType: .float32)

        eval(route.indices, route.scores)
        #expect(route.indices.asArray(Int32.self).map(Int.init) == [2])
        #expect(abs(route.scores.item(Float.self) - sigmoid(1) * 2) < 0.0001)
    }

    @Test("tiny model produces finite logits")
    func tinyModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = NemotronHModel(Self.smallConfig(pattern: "M*E-"))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)

            eval(logits)
            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("sanitizer packs expert projections and removes tied head")
    func sanitizerPacksExpertProjectionsAndRemovesTiedHead() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = NemotronHModel(Self.smallConfig(pattern: "E", tieWordEmbeddings: true))
            let sanitized = model.sanitize(weights: Self.sanitizerWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["backbone.layers.0.mixer.activation_scale"] == nil)
            #expect(sanitized["backbone.layers.0.mixer.experts.0.up_proj.weight"] == nil)

            let conv = try #require(sanitized["backbone.layers.0.mixer.conv1d.weight"])
            let upProjection = try #require(
                sanitized["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
            )
            let down = try #require(sanitized["backbone.layers.0.mixer.switch_mlp.fc2.weight"])

            eval(conv, upProjection, down)
            #expect(conv.shape == [2, 4, 3])
            #expect(upProjection.shape == [2, 3, 4])
            #expect(down.shape == [2, 4, 3])
            #expect(Array(upProjection.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [2, 2, 2, 2])
        }
    }

    private static func smallConfig(
        pattern: String,
        tieWordEmbeddings: Bool = false,
        nRoutedExperts: Int = 2,
        nGroup: Int = 1,
        topkGroup: Int = 1,
        routedScalingFactor: Float = 1
    ) -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 64,
            hiddenSize: 16,
            numHiddenLayers: pattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            mambaNumHeads: 2,
            mambaHeadDim: 4,
            ssmStateSize: 2,
            convKernel: 3,
            nGroups: 1,
            intermediateSize: 32,
            moeIntermediateSize: 8,
            moeSharedExpertIntermediateSize: 8,
            nRoutedExperts: nRoutedExperts,
            numExpertsPerTok: 1,
            hybridOverridePattern: pattern,
            tieWordEmbeddings: tieWordEmbeddings,
            nGroup: nGroup,
            topkGroup: topkGroup,
            routedScalingFactor: routedScalingFactor,
            timeStepLimitMin: 0.001,
            timeStepLimitMax: 100
        )
    }

    private static func configJSON(pattern: String) -> String {
        configJSON(patternJSON: "\"\(pattern)\"")
    }

    private static func configJSON(patternJSON: String) -> String {
        """
        {
            "vocab_size": 64,
            "hidden_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "mamba_num_heads": 2,
            "mamba_head_dim": 4,
            "ssm_state_size": 2,
            "conv_kernel": 3,
            "n_groups": 1,
            "intermediate_size": 32,
            "moe_intermediate_size": 8,
            "moe_shared_expert_intermediate_size": 8,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "hybrid_override_pattern": \(patternJSON),
            "time_step_limit": [0.001, 42.0]
        }
        """
    }

    private static let denseOnlyConfigJSON = """
    {
        "vocab_size": 64,
        "hidden_size": 16,
        "num_hidden_layers": 3,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
        "mamba_num_heads": 2,
        "mamba_head_dim": 4,
        "head_dim": 128,
        "ssm_state_size": 2,
        "conv_kernel": 3,
        "n_groups": 1,
        "intermediate_size": 32,
        "hybrid_override_pattern": "M*-"
    }
    """

    private static func sanitizerWeights() -> [String: MLXArray] {
        [
            "lm_head.weight": MLXArray.ones([4, 4]),
            "backbone.layers.0.mixer.conv1d.weight": MLXArray.zeros([2, 3, 4]),
            "backbone.layers.0.mixer.experts.0.up_proj.weight": filledArray(
                shape: [3, 4],
                value: 1
            ),
            "backbone.layers.0.mixer.experts.1.up_proj.weight": filledArray(
                shape: [3, 4],
                value: 2
            ),
            "backbone.layers.0.mixer.experts.0.down_proj.weight": filledArray(
                shape: [4, 3],
                value: 3
            ),
            "backbone.layers.0.mixer.experts.1.down_proj.weight": filledArray(
                shape: [4, 3],
                value: 4
            ),
            "backbone.layers.0.mixer.activation_scale": MLXArray.ones([1])
        ]
    }

    private static func filledArray(shape: [Int], value: Float) -> MLXArray {
        let count = shape.reduce(1, *)
        return MLXArray(Array(repeating: value, count: count)).reshaped(shape)
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }
}
