import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("ERNIE 4.5 MoE architecture")
struct Ernie45MoEArchitectureTests {
    @Test("decodes ERNIE 4.5 MoE configuration with routing metadata")
    func decodesConfigurationWithRoutingMetadata() throws {
        let config = try JSONDecoder.json5().decode(
            Ernie45MoEConfiguration.self,
            from: Data(Self.configJSON.utf8)
        )

        #expect(config.modelType == "ernie4_5_moe")
        #expect(config.hiddenSize == 2_560)
        #expect(config.intermediateSize == 12_288)
        #expect(config.maxPositionEmbeddings == 131_072)
        #expect(config.numAttentionHeads == 20)
        #expect(config.numKeyValueHeads == 4)
        #expect(config.headDim == nil)
        #expect(config.numHiddenLayers == 28)
        #expect(config.moeNumExperts == 64)
        #expect(config.moeLayerStartIndex.values == [1])
        #expect(config.moeIntermediateSize == 1_536)
        #expect(config.resolvedMoEIntermediateSize == 1_536)
        #expect(config.moeK == 6)
        #expect(config.moeLayerInterval == 1)
        #expect(config.moeNumSharedExperts == 2)
        #expect(config.moeLayerEndIndex == nil)
        #expect(config.moeGateActivation == .softmax)
    }

    @Test("builds attention and layer plans")
    func buildsAttentionAndLayerPlans() {
        let config = Self.smallConfig()
        let layout = Ernie45MoEAttentionLayout(config)
        let layerPlan = Ernie45MoELayerPlan(config)

        #expect(layout.hiddenSize == 16)
        #expect(layout.queryHeads == 4)
        #expect(layout.keyValueHeads == 2)
        #expect(layout.headSize == 4)
        #expect(layout.queryProjectionSize == 16)
        #expect(layout.keyValueProjectionSize == 8)
        #expect(layout.attentionScale == 0.5)
        #expect(!layerPlan.usesSparseExperts(layerIndex: 0))
        #expect(layerPlan.usesSparseExperts(layerIndex: 1))
    }

    @Test("routing plan normalizes selected experts")
    func routingPlanNormalizesSelectedExperts() {
        Device.withDefaultDevice(.cpu) {
            let plan = Ernie45MoERoutingPlan(
                Self.smallConfig(moeK: 2, moeGateActivation: .sigmoid)
            )
            let route = plan.route(MLXArray([Float(0), 1]).reshaped([1, 1, 2]))
            let scoreSum = route.scores.sum(axis: -1)
            eval(route.scores, route.indices, scoreSum)

            #expect(route.scores.shape == [1, 1, 2])
            #expect(route.indices.shape == [1, 1, 2])
            #expect(abs(scoreSum.item(Float.self) - 1) < 0.0001)
        }
    }

    @Test("constructs model with cache, adapters, and greedy fast path")
    func constructsModelWithCacheAdaptersAndGreedyFastPath() {
        let model = Ernie45MoEModel(Self.smallConfig(hiddenLayers: 2))
        let cache = model.newCache(parameters: nil)
        let loraTargets = model.loraLinearLayers()
        let _: any GreedyTokenModel = model

        #expect(model.vocabularySize == 64)
        #expect(model.kvHeads == [2, 2])
        #expect(cache.count == 2)
        #expect(loraTargets.count == 2)
        #expect(loraTargets[0].1 == ["q_proj", "v_proj"])
    }

    @Test("tiny tied model produces finite logits with dense and sparse layers")
    func tinyTiedModelProducesFiniteLogits() {
        Device.withDefaultDevice(.cpu) {
            let model = Ernie45MoEModel(Self.smallConfig(hiddenLayers: 2))
            let logits = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
            eval(logits)

            #expect(logits.shape == [1, 3, 64])
            #expect(all(isFinite(logits)).item(Bool.self))
        }
    }

    @Test("tiny untied model produces finite logits with cache")
    func tinyUntiedModelProducesFiniteLogitsWithCache() {
        Device.withDefaultDevice(.cpu) {
            let model = Ernie45MoEModel(
                Self.smallConfig(hiddenLayers: 2, tieWordEmbeddings: false)
            )
            let cache = model.newCache(parameters: nil)
            let prefill = model(MLXArray([1, 2]).reshaped(1, 2), cache: cache)
            let next = model(MLXArray([3]).reshaped(1, 1), cache: cache)
            eval(prefill, next)

            #expect(prefill.shape == [1, 2, 64])
            #expect(next.shape == [1, 1, 64])
            #expect(cache[0].offset == 3)
            #expect(all(isFinite(next)).item(Bool.self))
        }
    }

    @Test("sanitizer packs raw experts and strips unused tensors")
    func sanitizerPacksRawExpertsAndStripsUnusedTensors() throws {
        try Device.withDefaultDevice(.cpu) {
            let model = Ernie45MoEModel(Self.smallConfig(hiddenLayers: 1))
            let sanitized = model.sanitize(weights: Self.checkpointWeights())

            #expect(sanitized["lm_head.weight"] == nil)
            #expect(sanitized["lm_head.scales"] == nil)
            #expect(sanitized["lm_head.biases"] == nil)
            #expect(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"] == nil)
            #expect(sanitized["model.layers.0.mlp.experts.0.gate_proj.weight"] == nil)
            #expect(sanitized["model.layers.0.mlp.shared_experts.gate_proj.weight"] != nil)

            let gate = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.weight"])
            let gateScales = try #require(sanitized["model.layers.0.mlp.switch_mlp.gate_proj.scales"])
            let down = try #require(sanitized["model.layers.0.mlp.switch_mlp.down_proj.weight"])
            let upProjection = try #require(sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"])
            let bias = try #require(sanitized["model.layers.0.mlp.switch_mlp.up_proj.bias"])

            eval(gate, gateScales, down, upProjection, bias)
            #expect(gate.shape == [2, 2, 2])
            #expect(gateScales.shape == [2, 1])
            #expect(down.shape == [2, 2, 2])
            #expect(upProjection.shape == [2, 2, 2])
            #expect(bias.shape == [2, 2])
            #expect(Array(gate.asArray(Float.self).prefix(4)) == [1, 1, 1, 1])
            #expect(Array(upProjection.asArray(Float.self).suffix(4)) == [6, 6, 6, 6])
        }
    }

    private static func smallConfig(
        hiddenLayers: Int = 2,
        tieWordEmbeddings: Bool = true,
        moeK: Int = 1,
        moeGateActivation: Ernie45MoEGateActivation = .softmax
    ) -> Ernie45MoEConfiguration {
        Ernie45MoEConfiguration(
            hiddenSize: 16,
            intermediateSize: 32,
            maxPositionEmbeddings: 64,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            headDim: 4,
            numHiddenLayers: hiddenLayers,
            rmsNormEps: 1e-5,
            vocabularySize: 64,
            ropeTheta: 10_000,
            useBias: false,
            tieWordEmbeddings: tieWordEmbeddings,
            moeNumExperts: 2,
            moeLayerStartIndex: IntOrIntArray([1]),
            moeIntermediateSize: 8,
            moeK: moeK,
            moeLayerInterval: 1,
            moeNumSharedExperts: 1,
            moeGateActivation: moeGateActivation
        )
    }

    private static func checkpointWeights() -> [String: MLXArray] {
        var weights = baseCheckpointWeights()
        let projections = [
            ("gate_proj", Float(1)),
            ("down_proj", Float(3)),
            ("up_proj", Float(5))
        ]

        for expertIndex in 0 ..< 2 {
            for (name, baseValue) in projections {
                insertExpertWeights(
                    projection: name,
                    value: baseValue + Float(expertIndex),
                    expertIndex: expertIndex,
                    into: &weights
                )
            }
        }

        return weights
    }

    private static func baseCheckpointWeights() -> [String: MLXArray] {
        [
            "lm_head.weight": MLXArray.ones([2, 2]),
            "lm_head.scales": MLXArray.ones([1]),
            "lm_head.biases": MLXArray.ones([1]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.ones([2]),
            "model.layers.0.mlp.shared_experts.gate_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.0.mlp.shared_experts.down_proj.weight": MLXArray.ones([2, 2]),
            "model.layers.0.mlp.shared_experts.up_proj.weight": MLXArray.ones([2, 2])
        ]
    }

    private static func insertExpertWeights(
        projection: String,
        value: Float,
        expertIndex: Int,
        into weights: inout [String: MLXArray]
    ) {
        let prefix = "model.layers.0.mlp.experts.\(expertIndex).\(projection)"
        weights["\(prefix).weight"] = MLXArray([Float](repeating: value, count: 4))
            .reshaped([2, 2])
        weights["\(prefix).scales"] = MLXArray([Float](repeating: value, count: 1))
        weights["\(prefix).biases"] = MLXArray([Float](repeating: value, count: 1))
        weights["\(prefix).bias"] = MLXArray([Float](repeating: value, count: 2))
    }

    private static let configJSON = """
        {
            "model_type": "ernie4_5_moe",
            "hidden_size": 2560,
            "intermediate_size": 12288,
            "max_position_embeddings": 131072,
            "num_attention_heads": 20,
            "num_key_value_heads": 4,
            "head_dim": null,
            "num_hidden_layers": 28,
            "rms_norm_eps": 0.00001,
            "vocab_size": 103424,
            "rope_theta": 500000,
            "use_bias": false,
            "tie_word_embeddings": true,
            "moe_num_experts": 64,
            "moe_layer_start_index": 1,
            "moe_intermediate_size": 1536,
            "moe_k": 6,
            "moe_layer_interval": 1,
            "moe_num_shared_experts": 2,
            "moe_layer_end_index": null,
            "moe_gate_act": null
        }
        """
}
