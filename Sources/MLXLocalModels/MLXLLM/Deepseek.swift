import Foundation
import MLX
import MLXNN

internal struct DeepseekAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: DeepseekConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must be divisible by num_attention_heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "num_attention_heads must group num_key_value_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.keyValueHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal struct DeepseekRoPEPlan: Equatable, Sendable {
    internal let dimensions: Int
    internal let base: Float
    internal let scale: Float

    internal init(_ config: DeepseekConfiguration, dimensions: Int) {
        self.dimensions = dimensions
        self.base = config.ropeTheta
        self.scale = Self.scale(from: config.ropeScaling)
    }

    private static func scale(from ropeScaling: [String: StringOrNumber]?) -> Float {
        guard
            let ropeScaling,
            (ropeScaling["type"] ?? ropeScaling["rope_type"]) == .string("linear")
        else {
            return 1
        }

        guard let factor = ropeScaling["factor"]?.asFloat(), factor > 0 else {
            preconditionFailure("linear Deepseek rope_scaling requires a positive factor")
        }
        return 1 / factor
    }
}

internal struct DeepseekLayerPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let firstMoELayer: Int
    internal let moeLayerFrequency: Int
    internal let routedExperts: Int?

    internal init(_ config: DeepseekConfiguration) {
        precondition(config.hiddenLayers > 0, "num_hidden_layers must be positive")
        precondition(config.moeLayerFrequency > 0, "moe_layer_freq must be positive")
        self.layerCount = config.hiddenLayers
        self.firstMoELayer = config.firstDenseReplacementLayer
        self.moeLayerFrequency = config.moeLayerFrequency
        self.routedExperts = config.routedExperts
    }

    internal func usesMoE(at index: Int) -> Bool {
        routedExperts != nil && index >= firstMoELayer && index.isMultiple(of: moeLayerFrequency)
    }
}

internal struct DeepseekRoutingPlan: Equatable, Sendable {
    internal let routedExperts: Int
    internal let expertsPerToken: Int
    internal let normalizesSelectedProbabilities: Bool

    internal init(_ config: DeepseekConfiguration) {
        guard let routedExperts = config.routedExperts,
              let expertsPerToken = config.expertsPerToken
        else {
            preconditionFailure("Deepseek MoE requires routed experts and top-k count")
        }
        precondition(routedExperts > 0, "n_routed_experts must be positive")
        precondition(expertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            expertsPerToken <= routedExperts,
            "num_experts_per_tok cannot exceed n_routed_experts"
        )

        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.normalizesSelectedProbabilities = config.normTopKProbability
    }

    internal func route(_ logits: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let probabilities = softmax(logits, axis: -1, precise: true)
        let indices = argPartition(
            -probabilities,
            kth: expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<expertsPerToken]

        var scores = takeAlong(probabilities, indices, axis: -1)
        if normalizesSelectedProbabilities {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (indices, scores)
    }
}

private enum DeepseekExpertProjection: String, CaseIterable {
    case up = "up_proj"
    case down = "down_proj"
    case gate = "gate_proj"
}

internal struct DeepseekExpertPackingPlan: Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: DeepseekConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.routedExperts ?? 0
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        guard expertCount > 0 else { return weights }

        var packed = weights
        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in DeepseekExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.rawValue).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        stacked(tensors)
                }
            }
        }
        return packed
    }
}

private final class DeepseekAttention: Module {
    private let layout: DeepseekAttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: DeepseekConfiguration) {
        let layout = DeepseekAttentionLayout(config)
        let ropePlan = DeepseekRoPEPlan(config, dimensions: layout.headDimensions)
        self.layout = layout
        self.rope = RoPE(
            dimensions: ropePlan.dimensions,
            traditional: false,
            base: ropePlan.base,
            scale: ropePlan.scale
        )

        self._queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        self._keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: config.attentionBias
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)

        var queries = queryProjection(input)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        return outputProjection(
            attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: layout.attentionScale,
                mask: mask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, tokenCount, layout.queryProjectionSize)
        )
    }
}

private final class DeepseekMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProjection.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProjection.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    convenience init(_ config: DeepseekConfiguration, intermediateSize: Int? = nil) {
        self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: intermediateSize ?? config.intermediateSize
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class DeepseekMoEGate: Module {
    private let routingPlan: DeepseekRoutingPlan

    @ModuleInfo(key: "weight") private var weight: MLXArray

    init(_ config: DeepseekConfiguration) {
        self.routingPlan = DeepseekRoutingPlan(config)
        self._weight.wrappedValue = zeros([routingPlan.routedExperts, config.hiddenSize])
    }

    func callAsFunction(_ input: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        routingPlan.route(input.matmul(weight.T))
    }
}

private final class DeepseekMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") private var gate: DeepseekMoEGate
    @ModuleInfo(key: "shared_experts") private var sharedExperts: DeepseekMLP?

    init(_ config: DeepseekConfiguration) {
        let routingPlan = DeepseekRoutingPlan(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: routingPlan.routedExperts,
            bias: false
        )
        self._gate.wrappedValue = DeepseekMoEGate(config)

        if let sharedExpertCount = config.sharedExperts, sharedExpertCount > 0 {
            self._sharedExperts.wrappedValue = DeepseekMLP(
                config,
                intermediateSize: config.moeIntermediateSize * sharedExpertCount
            )
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let route = gate(input)
        var output = switchMLP(input, route.indices)
        output = (output * route.scores.asType(output.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            output = output + sharedExperts(input)
        }
        return output
    }
}

private final class DeepseekBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: DeepseekAttention
    @ModuleInfo(key: "mlp") private var feedForward: UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: DeepseekConfiguration, layerIndex: Int) {
        self._attention.wrappedValue = DeepseekAttention(config)
        if DeepseekLayerPlan(config).usesMoE(at: layerIndex) {
            self._feedForward.wrappedValue = DeepseekMoE(config)
        } else {
            self._feedForward.wrappedValue = DeepseekMLP(config)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(inputLayerNorm(input), mask: mask, cache: cache)
        let hiddenStates = input + attentionOutput
        return hiddenStates + feedForward(postAttentionLayerNorm(hiddenStates))
    }
}

private final class DeepseekBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [DeepseekBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: DeepseekConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { index in
            DeepseekBlock(config, layerIndex: index)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[index])
        }
        return norm(hiddenStates)
    }
}

internal final class DeepseekModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    internal let modelType: String
    private let configuration: DeepseekConfiguration

    @ModuleInfo(key: "model") private var model: DeepseekBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: DeepseekConfiguration) {
        self.configuration = configuration
        self.modelType = configuration.modelType
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(repeating: configuration.keyValueHeads, count: configuration.hiddenLayers)
        self._model.wrappedValue = DeepseekBackbone(configuration)

        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        greedyTokenOutput(
            logits: logits(
                from: lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
            ),
            state: state
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights
        sanitized = DeepseekExpertPackingPlan(configuration).pack(sanitized)
        sanitized = sanitized.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return sanitized
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) } ?? model.embeddings.asLinear(hiddenStates)
    }
}

internal struct DeepseekConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var keyValueHeads: Int
    var sharedExperts: Int?
    var routedExperts: Int?
    var expertsPerToken: Int?
    var moeLayerFrequency: Int
    var firstDenseReplacementLayer: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool
    var normTopKProbability: Bool
    var scoringFunction: String
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "deepseek",
        vocabularySize: Int = 102_400,
        hiddenSize: Int = 4_096,
        intermediateSize: Int = 11_008,
        moeIntermediateSize: Int = 1_407,
        hiddenLayers: Int = 30,
        attentionHeads: Int = 32,
        keyValueHeads: Int? = nil,
        sharedExperts: Int? = nil,
        routedExperts: Int? = nil,
        expertsPerToken: Int? = nil,
        moeLayerFrequency: Int = 1,
        firstDenseReplacementLayer: Int = 0,
        maxPositionEmbeddings: Int = 2_048,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        attentionBias: Bool = false,
        normTopKProbability: Bool = false,
        scoringFunction: String = "softmax",
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.sharedExperts = sharedExperts
        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.moeLayerFrequency = moeLayerFrequency
        self.firstDenseReplacementLayer = firstDenseReplacementLayer
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.attentionBias = attentionBias
        self.normTopKProbability = normTopKProbability
        self.scoringFunction = scoringFunction
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case sharedExperts = "n_shared_experts"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeLayerFrequency = "moe_layer_freq"
        case firstDenseReplacementLayer = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case normTopKProbability = "norm_topk_prob"
        case scoringFunction = "scoring_func"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
            ?? 32
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "deepseek",
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 102_400,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4_096,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 11_008,
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? 1_407,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30,
            attentionHeads: attentionHeads,
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? attentionHeads,
            sharedExperts: try container.decodeIfPresent(Int.self, forKey: .sharedExperts),
            routedExperts: try container.decodeIfPresent(Int.self, forKey: .routedExperts),
            expertsPerToken: try container.decodeIfPresent(Int.self, forKey: .expertsPerToken),
            moeLayerFrequency: try container.decodeIfPresent(
                Int.self,
                forKey: .moeLayerFrequency
            ) ?? 1,
            firstDenseReplacementLayer: try container.decodeIfPresent(
                Int.self,
                forKey: .firstDenseReplacementLayer
            ) ?? 0,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 2_048,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            normTopKProbability: try container.decodeIfPresent(
                Bool.self,
                forKey: .normTopKProbability
            ) ?? false,
            scoringFunction: try container.decodeIfPresent(String.self, forKey: .scoringFunction)
                ?? "softmax",
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

extension DeepseekModel: LoRAModel {
    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
