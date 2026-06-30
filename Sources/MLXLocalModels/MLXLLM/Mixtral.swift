import Foundation
import MLX
import MLXNN

internal struct MixtralAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: MixtralConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.attentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.keyValueHeads > 0, "num_key_value_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "hidden_size must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.keyValueHeads),
            "attention heads must group key-value heads"
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

internal struct MixtralRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int

    internal init(_ config: MixtralConfiguration) {
        precondition(config.expertCount > 0, "num_local_experts must be positive")
        precondition(config.expertsPerToken > 0, "num_experts_per_tok must be positive")
        precondition(
            config.expertsPerToken <= config.expertCount,
            "num_experts_per_tok cannot exceed num_local_experts"
        )

        self.expertCount = config.expertCount
        self.selectedExpertCount = config.expertsPerToken
    }

    internal func route(_ logits: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        let indices = argPartition(
            -logits,
            kth: selectedExpertCount - 1,
            axis: -1
        )[.ellipsis, ..<selectedExpertCount]
        let selectedLogits = takeAlong(logits, indices, axis: -1)
        return (softmax(selectedLogits, axis: -1, precise: true), indices)
    }
}

private enum MixtralExpertProjection: String, CaseIterable {
    case w1 = "gate_proj"
    case w2 = "down_proj"
    case w3 = "up_proj"
}

internal struct MixtralExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: MixtralConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.expertCount
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights
        guard packed["model.layers.0.block_sparse_moe.experts.0.w1.weight"] != nil else {
            return packed
        }

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).block_sparse_moe"
            for projection in MixtralExpertProjection.allCases {
                for tensorName in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(tensorName)"
                    guard packed[firstKey] != nil else { continue }

                    let tensors = (0 ..< expertCount).map { expertIndex in
                        packed.removeValue(
                            forKey: "\(prefix).experts.\(expertIndex).\(projection).\(tensorName)"
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

private final class MixtralAttention: Module {
    private let layout: MixtralAttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: MixtralConfiguration) {
        let layout = MixtralAttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.headDimensions,
            traditional: config.ropeTraditional,
            base: config.ropeTheta,
            scale: config.ropeScale
        )

        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.queryProjectionSize,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            layout.keyValueProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            config.hiddenSize,
            bias: false
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

private final class MixtralSparseMoE: Module, UnaryLayer {
    private let routingPlan: MixtralRoutingPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU

    init(_ config: MixtralConfiguration) {
        self.routingPlan = MixtralRoutingPlan(config)
        _gate.wrappedValue = Linear(config.hiddenSize, config.expertCount, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.expertCount
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routed = routingPlan.route(gate(input))
        let expertOutput = switchMLP(input, routed.indices)
        return (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

private final class MixtralBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MixtralAttention
    @ModuleInfo(key: "block_sparse_moe") var feedForward: MixtralSparseMoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: MixtralConfiguration) {
        _attention.wrappedValue = MixtralAttention(config)
        _feedForward.wrappedValue = MixtralSparseMoE(config)
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
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = input + attention(inputLayerNorm(input), mask: mask, cache: cache)
        return attended + feedForward(postAttentionLayerNorm(attended))
    }
}

private final class MixtralBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [MixtralBlock]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ config: MixtralConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.hiddenLayers).map { _ in MixtralBlock(config) }
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

internal final class MixtralModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: MixtralConfiguration

    @ModuleInfo(key: "model") private var model: MixtralBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: MixtralConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.keyValueHeads, count: config.hiddenLayers)
        _model.wrappedValue = MixtralBackbone(config)

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
        var sanitized = MixtralExpertPackingPlan(configuration).pack(weights)
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        return sanitized.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
    }
}

internal struct MixtralConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var keyValueHeads: Int
    var expertCount: Int
    var expertsPerToken: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    internal var ropeScale: Float {
        guard let ropeScaling, ropeScaling["type"] == .string("linear") else {
            return 1
        }
        guard let factor = ropeScaling["factor"]?.asFloat(), factor > 0 else {
            preconditionFailure("mixtral rope_scaling.factor must be positive")
        }
        return 1 / factor
    }

    internal init(
        modelType: String = "mixtral",
        hiddenSize: Int,
        hiddenLayers: Int,
        intermediateSize: Int,
        attentionHeads: Int,
        keyValueHeads: Int? = nil,
        expertCount: Int,
        expertsPerToken: Int,
        rmsNormEps: Float,
        vocabularySize: Int,
        ropeTheta: Float = 1_000_000,
        ropeTraditional: Bool = false,
        ropeScaling: [String: StringOrNumber]? = nil,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.expertCount = expertCount
        self.expertsPerToken = expertsPerToken
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.ropeTraditional = ropeTraditional
        self.ropeScaling = ropeScaling
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case expertCount = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
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
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "mixtral",
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            attentionHeads: attentionHeads,
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads),
            expertCount: try container.decode(Int.self, forKey: .expertCount),
            expertsPerToken: try container.decode(Int.self, forKey: .expertsPerToken),
            rmsNormEps: try container.decode(Float.self, forKey: .rmsNormEps),
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000,
            ropeTraditional: try container.decodeIfPresent(Bool.self, forKey: .ropeTraditional)
                ?? false,
            ropeScaling: ropeScaling,
            tieWordEmbeddings: try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
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
                    debugDescription: "mixtral rope_scaling must be linear with a numeric factor"
                )
            )
        }
    }
}

extension MixtralModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
