import Foundation
import MLX
import MLXNN

internal struct Qwen2MoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: Qwen2MoEConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.kvHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "attention heads must group key-value heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var queryProjectionSize: Int { attentionHeads * headDimensions }
    internal var keyValueProjectionSize: Int { keyValueHeads * headDimensions }
}

internal struct Qwen2MoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let normalizesSelectedProbabilities: Bool

    internal init(_ config: Qwen2MoEConfiguration) {
        precondition(config.numExperts > 0, "num_experts must be positive")
        precondition(config.numExpertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            config.numExpertsPerToken <= config.numExperts,
            "num_experts_per_tok cannot exceed num_experts"
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

private enum Qwen2MoEExpertProjection: String, CaseIterable {
    case up = "up_proj"
    case down = "down_proj"
    case gate = "gate_proj"
}

internal struct Qwen2MoEExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: Qwen2MoEConfiguration) {
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
            for projection in Qwen2MoEExpertProjection.allCases {
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

internal final class Qwen2MoEAttention: Module {
    let layout: Qwen2MoEAttentionLayout

    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    let rope: RoPE

    init(_ config: Qwen2MoEConfiguration) {
        self.layout = Qwen2MoEAttentionLayout(config)

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: true
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: false
        )

        self.rope = RoPE(
            dimensions: layout.headDimensions,
            traditional: config.ropeTraditional,
            base: config.ropeTheta,
            scale: config.ropeScale
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        var queries = queryProjection(x)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(output)
    }
}

internal final class Qwen2MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class Qwen2MoESparseBlock: Module, UnaryLayer {
    let routingPlan: Qwen2MoERoutingPlan

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: Qwen2MoEMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ config: Qwen2MoEConfiguration) {
        self.routingPlan = Qwen2MoERoutingPlan(config)

        _gate.wrappedValue = Linear(config.hiddenSize, routingPlan.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: routingPlan.expertCount
        )
        _sharedExpert.wrappedValue = Qwen2MoEMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let routed = routingPlan.route(gate(x))
        let expertOutput = switchMLP(x, routed.indices)
        let routedOutput = (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
        let sharedOutput = MLX.sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        return routedOutput + sharedOutput
    }
}

internal final class Qwen2MoEBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2MoEAttention
    @ModuleInfo(key: "mlp") var feedForward: Qwen2MoESparseBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: Qwen2MoEConfiguration) {
        _attention.wrappedValue = Qwen2MoEAttention(config)
        _feedForward.wrappedValue = Qwen2MoESparseBlock(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attentionOutput = attention(inputLayerNorm(x), mask: mask, cache: cache)
        let hidden = x + attentionOutput
        return hidden + feedForward(postAttentionLayerNorm(hidden))
    }
}

internal final class Qwen2MoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen2MoEBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: Qwen2MoEConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in Qwen2MoEBlock(config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = embedTokens(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache)

        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }

        return norm(hidden)
    }
}

internal final class Qwen2MoEModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: Qwen2MoEConfiguration

    @ModuleInfo(key: "model") var model: Qwen2MoEBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: Qwen2MoEConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _model.wrappedValue = Qwen2MoEBackbone(config)

        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var output = model(inputs, cache: cache)
        if let lmHead {
            output = lmHead(output)
        } else {
            output = model.embedTokens.asLinear(output)
        }
        return output
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        var logits = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        if let lmHead {
            logits = lmHead(logits)
        } else {
            logits = model.embedTokens.asLinear(logits)
        }
        return greedyTokenOutput(logits: logits, state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = Qwen2MoEExpertPackingPlan(configuration).pack(weights)

        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

internal struct Qwen2MoEConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var numExpertsPerToken: Int
    var numExperts: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var normTopkProb: Bool

    internal var ropeScale: Float {
        guard let ropeScaling, ropeScaling["type"] == .string("linear") else {
            return 1
        }
        guard let factor = ropeScaling["factor"]?.asFloat(), factor != 0 else {
            preconditionFailure("rope_scaling.factor must be a non-zero number")
        }
        return 1 / factor
    }

    internal init(
        modelType: String = "qwen2_moe",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        numExpertsPerToken: Int,
        numExperts: Int,
        moeIntermediateSize: Int,
        sharedExpertIntermediateSize: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        kvHeads: Int? = nil,
        ropeTheta: Float = 1_000_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false,
        normTopkProb: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.numExpertsPerToken = numExpertsPerToken
        self.numExperts = numExperts
        self.moeIntermediateSize = moeIntermediateSize
        self.sharedExpertIntermediateSize = sharedExpertIntermediateSize
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.kvHeads = kvHeads ?? attentionHeads
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
        self.normTopkProb = normTopkProb
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case numExpertsPerToken = "num_experts_per_tok"
        case numExperts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case normTopkProb = "norm_topk_prob"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try container.decode(Int.self, forKey: .attentionHeads)
        let ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling
        )
        try Self.validate(ropeScaling)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen2_moe",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            numExpertsPerToken: try container.decode(Int.self, forKey: .numExpertsPerToken),
            numExperts: try container.decode(Int.self, forKey: .numExperts),
            moeIntermediateSize: try container.decode(Int.self, forKey: .moeIntermediateSize),
            sharedExpertIntermediateSize: try container.decode(
                Int.self,
                forKey: .sharedExpertIntermediateSize
            ),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_000_000,
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? false,
            normTopkProb: try container.decodeIfPresent(Bool.self, forKey: .normTopkProb)
                ?? false
        )
    }

    private static func validate(_ ropeScaling: [String: StringOrNumber]?) throws {
        guard let ropeScaling else { return }
        guard ropeScaling["type"] == .string("linear"),
              ropeScaling["factor"]?.asFloat() != nil else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: [],
                    debugDescription: "qwen2_moe rope_scaling must be linear with a numeric factor"
                )
            )
        }
    }
}

extension Qwen2MoEModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
