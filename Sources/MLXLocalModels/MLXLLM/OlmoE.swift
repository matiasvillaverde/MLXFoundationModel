import Foundation
import MLX
import MLXNN

// MARK: - Plans

internal struct OlmoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: OlmoEConfiguration) {
        let headDimensions = config.headDimensions ?? (config.hiddenSize / config.attentionHeads)
        precondition(config.attentionHeads > 0, "OLMoE attention heads must be positive")
        precondition(config.kvHeads > 0, "OLMoE KV heads must be positive")
        precondition(headDimensions > 0, "OLMoE head dimensions must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "OLMoE attention heads must group KV heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = headDimensions
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal struct OlmoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let normalizesSelectedProbabilities: Bool

    internal init(_ config: OlmoEConfiguration) {
        precondition(config.numExperts > 0, "OLMoE expert count must be positive")
        precondition(config.numExpertsPerToken > 0, "OLMoE selected expert count must be positive")
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "OLMoE cannot select more experts than it owns"
        )

        self.expertCount = config.numExperts
        self.selectedExpertCount = config.numExpertsPerToken
        self.normalizesSelectedProbabilities = config.normTopkProb
    }

    internal func route(_ logits: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        let probabilities = MLX.softmax(logits, axis: -1, precise: true)
        let indices = MLX.argPartition(
            -probabilities,
            kth: selectedExpertCount - 1,
            axis: -1
        )[.ellipsis, ..<selectedExpertCount]

        var scores = MLX.takeAlong(probabilities, indices, axis: -1)
        if normalizesSelectedProbabilities {
            scores = scores / MLX.sum(scores, axis: -1, keepDims: true)
        }
        return (scores, indices)
    }
}

private enum OlmoEExpertProjection: String, CaseIterable {
    case up = "up_proj"
    case down = "down_proj"
    case gate = "gate_proj"
}

internal struct OlmoEExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: OlmoEConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.numExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights
        guard packed["model.layers.0.mlp.experts.0.up_proj.weight"] != nil else {
            return packed
        }

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in OlmoEExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection.rawValue).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection.rawValue).\(tensorName)"
                        )!
                    }
                    packed["\(prefix).switch_mlp.\(projection.rawValue).\(tensorName)"] =
                        MLX.stacked(tensors)
                }
            }
        }

        return packed
    }
}

// MARK: - Model Components

internal final class OlmoEAttention: Module {
    let layout: OlmoEAttentionLayout

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ config: OlmoEConfiguration) {
        self.layout = OlmoEAttentionLayout(config)

        _qProj.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        _kProj.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _vProj.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _oProj.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: config.attentionBias
        )

        _qNorm.wrappedValue = RMSNorm(dimensions: layout.queryProjectionSize, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: layout.keyValueProjectionSize, eps: config.rmsNormEps)

        self.rope = initializeRope(
            dims: layout.headDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        var queryStates = qNorm(qProj(x))
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keyStates = kNorm(kProj(x))
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let valueStates = vProj(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        queryStates = applyRotaryPosition(rope, to: queryStates, cache: cache)
        keyStates = applyRotaryPosition(rope, to: keyStates, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queryStates,
            keys: keyStates,
            values: valueStates,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return oProj(output)
    }
}

internal final class OlmoESparseMoeBlock: Module, UnaryLayer {
    let routingPlan: OlmoERoutingPlan

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ config: OlmoEConfiguration) {
        self.routingPlan = OlmoERoutingPlan(config)

        _gate.wrappedValue = Linear(config.hiddenSize, routingPlan.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: routingPlan.expertCount,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = routingPlan.route(gate(x))
        let expertOutput = switchMLP(x, routed.indices)
        return (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

internal final class OlmoEBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: OlmoEAttention
    @ModuleInfo(key: "mlp") fileprivate var feedForward: OlmoESparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: OlmoEConfiguration) {
        _attention.wrappedValue = OlmoEAttention(config)
        _feedForward.wrappedValue = OlmoESparseMoeBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hidden = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        hidden = hidden + feedForward(postAttentionLayerNorm(hidden))
        return hidden
    }
}

internal final class OlmoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [OlmoEBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: OlmoEConfiguration) {
        precondition(config.vocabularySize > 0, "OLMoE vocabulary size must be positive")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in OlmoEBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (layerIndex, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[layerIndex])
        }

        return norm(hidden)
    }
}

internal final class OlmoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") private var model: OlmoEBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    private let configuration: OlmoEConfiguration
    let modelType: String

    internal init(_ config: OlmoEConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        self.modelType = config.modelType
        self._model.wrappedValue = OlmoEBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenStates), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.hiddenLayers).map { _ in KVCacheSimple() }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }
        return OlmoEExpertPackingPlan(configuration).pack(sanitized)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}

// MARK: - Configuration

internal struct OlmoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var headDimensions: Int?
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var numExperts: Int
    var numExpertsPerToken: Int
    var normTopkProb: Bool

    internal init(
        modelType: String = "olmoe",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        headDimensions: Int? = nil,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        maxPositionEmbeddings: Int? = nil,
        ropeTheta: Float = 10_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = true,
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        numExperts: Int,
        numExpertsPerToken: Int,
        normTopkProb: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.headDimensions = headDimensions
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.numExperts = numExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.normTopkProb = normTopkProb
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "olmoe",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            headDimensions: try container.decodeIfPresent(Int.self, forKey: .headDimensions),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false
        )
    }
}

// MARK: - LoRA

extension OlmoEModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
