import Darwin
import MLX
import MLXFast
import MLXNN

internal struct DbrxConfiguration: Codable, Equatable, Sendable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var hiddenLayers: Int
    internal var attentionHeads: Int
    internal var attentionConfig: AttentionConfig
    internal var feedForwardConfig: FeedForwardConfig
    internal var layerNormEpsilon: Float
    internal var tieWordEmbeddings: Bool

    internal struct AttentionConfig: Codable, Equatable, Sendable {
        internal var keyValueHeads: Int
        internal var clipQKV: Float
        internal var ropeTheta: Float

        internal init(keyValueHeads: Int, clipQKV: Float = 8, ropeTheta: Float = 500_000) {
            self.keyValueHeads = keyValueHeads
            self.clipQKV = clipQKV
            self.ropeTheta = ropeTheta
        }

        enum CodingKeys: String, CodingKey {
            case keyValueHeads = "kv_n_heads"
            case clipQKV = "clip_qkv"
            case ropeTheta = "rope_theta"
        }
    }

    internal struct FeedForwardConfig: Codable, Equatable, Sendable {
        internal var hiddenSize: Int
        internal var expertCount: Int
        internal var expertsPerToken: Int

        internal init(hiddenSize: Int, expertCount: Int, expertsPerToken: Int) {
            self.hiddenSize = hiddenSize
            self.expertCount = expertCount
            self.expertsPerToken = expertsPerToken
        }

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "ffn_hidden_size"
            case expertCount = "moe_num_experts"
            case expertsPerToken = "moe_top_k"
        }
    }

    internal init(
        modelType: String = "dbrx",
        vocabularySize: Int,
        hiddenSize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        attentionConfig: AttentionConfig,
        feedForwardConfig: FeedForwardConfig,
        layerNormEpsilon: Float = 1e-5,
        tieWordEmbeddings: Bool = false
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.attentionConfig = attentionConfig
        self.feedForwardConfig = feedForwardConfig
        self.layerNormEpsilon = layerNormEpsilon
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "d_model"
        case hiddenLayers = "n_layers"
        case attentionHeads = "n_heads"
        case attentionConfig = "attn_config"
        case feedForwardConfig = "ffn_config"
        case layerNormEpsilon = "layer_norm_epsilon"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "dbrx",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            attentionConfig: try container.decode(
                AttentionConfig.self,
                forKey: .attentionConfig
            ),
            feedForwardConfig: try container.decode(
                FeedForwardConfig.self,
                forKey: .feedForwardConfig
            ),
            layerNormEpsilon: try container.decodeIfPresent(
                Float.self,
                forKey: .layerNormEpsilon
            ) ?? 1e-5,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

internal struct DbrxAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let packedProjectionSize: Int
    internal let clipQKV: Float
    internal let attentionScale: Float

    internal init(_ config: DbrxConfiguration) {
        precondition(config.hiddenSize > 0, "d_model must be positive")
        precondition(config.attentionHeads > 0, "n_heads must be positive")
        precondition(config.attentionConfig.keyValueHeads > 0, "kv_n_heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "d_model must divide evenly across n_heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.attentionConfig.keyValueHeads),
            "n_heads must be a multiple of kv_n_heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.keyValueHeads = config.attentionConfig.keyValueHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.queryProjectionSize = config.hiddenSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.packedProjectionSize = queryProjectionSize + 2 * keyValueProjectionSize
        self.clipQKV = config.attentionConfig.clipQKV
        self.attentionScale = powf(Float(headSize), -0.5)
    }
}

internal struct DbrxRoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let expertsPerToken: Int

    internal init(_ config: DbrxConfiguration) {
        precondition(config.feedForwardConfig.expertCount > 0, "moe_num_experts must be positive")
        precondition(config.feedForwardConfig.expertsPerToken > 0, "moe_top_k must be positive")
        precondition(
            config.feedForwardConfig.expertsPerToken <= config.feedForwardConfig.expertCount,
            "moe_top_k cannot exceed moe_num_experts"
        )

        self.expertCount = config.feedForwardConfig.expertCount
        self.expertsPerToken = config.feedForwardConfig.expertsPerToken
    }

    internal func route(_ logits: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        let indices = argPartition(
            -logits,
            kth: expertsPerToken - 1,
            axis: -1
        )[.ellipsis, ..<expertsPerToken]
        let selectedLogits = takeAlong(logits, indices, axis: -1)
        return (softmax(selectedLogits, axis: -1, precise: true), indices)
    }
}

private final class DbrxLayerNorm: Module {
    private let epsilon: Float
    @ParameterInfo(key: "weight") private var weight: MLXArray

    init(dimensions: Int, epsilon: Float) {
        self.epsilon = epsilon
        self._weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let average = mean(input, axis: -1, keepDims: true)
        let centered = input - average
        let variance = mean(centered.square(), axis: -1, keepDims: true)
        return weight * centered * (variance + epsilon).rsqrt()
    }
}

private final class DbrxAttention: Module {
    private let layout: DbrxAttentionLayout
    private let rope: RoPE?

    @ModuleInfo(key: "Wqkv") private var queryKeyValueProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear

    init(_ config: DbrxConfiguration) {
        self.layout = DbrxAttentionLayout(config)
        self.rope = layout.headSize > 1
            ? RoPE(
                dimensions: layout.headSize,
                traditional: false,
                base: config.attentionConfig.ropeTheta
            )
            : nil
        self._queryKeyValueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.packedProjectionSize,
            bias: false
        )
        self._outputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.hiddenSize,
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
        let clipped = clip(
            queryKeyValueProjection(input),
            min: -layout.clipQKV,
            max: layout.clipQKV
        )
        let qkv = split(
            clipped,
            indices: [
                layout.queryProjectionSize,
                layout.queryProjectionSize + layout.keyValueProjectionSize
            ],
            axis: -1
        )

        var queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache, let rope {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else if let rope {
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

private final class DbrxRouter: Module {
    @ModuleInfo(key: "layer") private var layer: Linear

    init(_ config: DbrxConfiguration) {
        self._layer.wrappedValue = Linear(
            config.hiddenSize,
            config.feedForwardConfig.expertCount,
            bias: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        layer(input)
    }
}

private final class DbrxSparseMoE: Module, UnaryLayer {
    private let routingPlan: DbrxRoutingPlan

    @ModuleInfo(key: "router") private var router: DbrxRouter
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU

    init(_ config: DbrxConfiguration) {
        self.routingPlan = DbrxRoutingPlan(config)
        self._router.wrappedValue = DbrxRouter(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.feedForwardConfig.hiddenSize,
            numExperts: config.feedForwardConfig.expertCount
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let routed = routingPlan.route(router(input))
        let expertOutput = switchMLP(input, routed.indices)
        return (expertOutput * routed.scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

private final class DbrxNormAttentionNorm: Module {
    @ModuleInfo(key: "norm_1") private var inputNorm: DbrxLayerNorm
    @ModuleInfo(key: "norm_2") private var postAttentionNorm: DbrxLayerNorm
    @ModuleInfo(key: "attn") fileprivate var attention: DbrxAttention

    init(_ config: DbrxConfiguration) {
        self._inputNorm.wrappedValue = DbrxLayerNorm(
            dimensions: config.hiddenSize,
            epsilon: config.layerNormEpsilon
        )
        self._postAttentionNorm.wrappedValue = DbrxLayerNorm(
            dimensions: config.hiddenSize,
            epsilon: config.layerNormEpsilon
        )
        self._attention.wrappedValue = DbrxAttention(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> (residual: MLXArray, normalized: MLXArray) {
        let attended = input + attention(inputNorm(input), mask: mask, cache: cache)
        return (attended, postAttentionNorm(attended))
    }
}

private final class DbrxDecoderLayer: Module {
    @ModuleInfo(key: "norm_attn_norm") fileprivate var normAttentionNorm: DbrxNormAttentionNorm
    @ModuleInfo(key: "ffn") private var feedForward: DbrxSparseMoE

    init(_ config: DbrxConfiguration) {
        self._normAttentionNorm.wrappedValue = DbrxNormAttentionNorm(config)
        self._feedForward.wrappedValue = DbrxSparseMoE(config)
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let result = normAttentionNorm(input, mask: mask, cache: cache)
        return result.residual + feedForward(result.normalized)
    }
}

private final class DbrxBackbone: Module {
    @ModuleInfo(key: "wte") fileprivate var tokenEmbeddings: Embedding
    @ModuleInfo(key: "blocks") fileprivate var layers: [DbrxDecoderLayer]
    @ModuleInfo(key: "norm_f") private var finalNorm: DbrxLayerNorm

    init(_ config: DbrxConfiguration) {
        self._tokenEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            DbrxDecoderLayer(config)
        }
        self._finalNorm.wrappedValue = DbrxLayerNorm(
            dimensions: config.hiddenSize,
            epsilon: config.layerNormEpsilon
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hidden = tokenEmbeddings(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache)

        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[index])
        }

        return finalNorm(hidden)
    }
}

internal final class DbrxModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: DbrxConfiguration
    @ModuleInfo(key: "transformer") private var transformer: DbrxBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: DbrxConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = Array(
            repeating: configuration.attentionConfig.keyValueHeads,
            count: configuration.hiddenLayers
        )
        self._transformer.wrappedValue = DbrxBackbone(configuration)

        if !configuration.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
        }
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(from: transformer(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hidden = lastTokenHiddenState(transformer(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = DbrxExpertPackingPlan(configuration).pack(weights)
        if configuration.tieWordEmbeddings {
            packed["lm_head.weight"] = nil
        }
        return packed
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return transformer.tokenEmbeddings.asLinear(hiddenStates)
    }
}

internal struct DbrxExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: DbrxConfiguration) {
        self.layerCount = config.hiddenLayers
        self.expertCount = config.feedForwardConfig.expertCount
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for (key, tensor) in weights {
            guard let packedExpert = DbrxPackedExpertTensor(key) else {
                continue
            }
            let experts = split(tensor, parts: expertCount, axis: 0).map {
                packedExpert.projection.requiresTranspose ? $0.T : $0
            }
            packed[key] = nil
            packed[packedExpert.destinationKey] = stacked(experts)
        }

        return packed
    }
}

private struct DbrxPackedExpertTensor {
    internal let prefix: String
    internal let projection: Projection

    internal enum Projection: String {
        case w1
        case v1
        case w2

        internal var destination: String {
            switch self {
            case .w1:
                "gate_proj"
            case .v1:
                "up_proj"
            case .w2:
                "down_proj"
            }
        }

        internal var requiresTranspose: Bool {
            self == .w2
        }
    }

    internal init?(_ key: String) {
        guard let range = key.range(of: ".experts.mlp.") else {
            return nil
        }
        let suffix = String(key[range.upperBound...])
        guard let projection = Projection(rawValue: suffix) else {
            return nil
        }
        self.prefix = String(key[..<range.lowerBound])
        self.projection = projection
    }

    internal var destinationKey: String {
        "\(prefix).switch_mlp.\(projection.destination).weight"
    }
}

extension DbrxModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.normAttentionNorm.attention, ["Wqkv", "out_proj"]) }
    }
}
