import Foundation
import MLX
import MLXNN

// MARK: - Configuration

internal struct GPT2Configuration: Codable, Sendable, Equatable {
    var modelType: String
    var contextLength: Int
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var maxPositionEmbeddings: Int
    var layerNormEps: Float
    var vocabularySize: Int
    var innerSize: Int?
    var activationFunction: String

    internal var feedForwardSize: Int {
        innerSize ?? 4 * hiddenSize
    }

    internal init(
        modelType: String = "gpt2",
        contextLength: Int = 1_024,
        hiddenSize: Int = 768,
        attentionHeads: Int = 12,
        hiddenLayers: Int = 12,
        maxPositionEmbeddings: Int = 1_024,
        layerNormEps: Float = 1e-5,
        vocabularySize: Int = 50_257,
        innerSize: Int? = nil,
        activationFunction: String = "gelu_new"
    ) {
        self.modelType = modelType
        self.contextLength = contextLength
        self.hiddenSize = hiddenSize
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.layerNormEps = layerNormEps
        self.vocabularySize = vocabularySize
        self.innerSize = innerSize
        self.activationFunction = activationFunction
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case contextLength = "n_ctx"
        case hiddenSize = "n_embd"
        case attentionHeads = "n_head"
        case hiddenLayers = "n_layer"
        case maxPositionEmbeddings = "n_positions"
        case layerNormEps = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case innerSize = "n_inner"
        case activationFunction = "activation_function"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let maxPositionEmbeddings = try container.decodeIfPresent(
            Int.self,
            forKey: .maxPositionEmbeddings
        ) ?? 1_024

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "gpt2",
            contextLength: try container.decodeIfPresent(Int.self, forKey: .contextLength)
                ?? maxPositionEmbeddings,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12,
            hiddenLayers: try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12,
            maxPositionEmbeddings: maxPositionEmbeddings,
            layerNormEps: try container.decodeIfPresent(Float.self, forKey: .layerNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 50_257,
            innerSize: try container.decodeIfPresent(Int.self, forKey: .innerSize),
            activationFunction: try container.decodeIfPresent(
                String.self,
                forKey: .activationFunction
            ) ?? "gelu_new"
        )
    }
}

// MARK: - Plans

internal struct GPT2AttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let headDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: GPT2Configuration) {
        precondition(config.hiddenSize > 0, "GPT-2 hidden size must be positive")
        precondition(config.attentionHeads > 0, "GPT-2 attention head count must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "GPT-2 hidden size must divide evenly across attention heads"
        )

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.attentionScale = pow(Float(headDimensions), -0.5)
    }

    internal var combinedProjectionSize: Int {
        3 * hiddenSize
    }
}

private enum GPT2RawLinearProjection: String, CaseIterable {
    case attentionQKV = "attn.c_attn"
    case attentionOutput = "attn.c_proj"
    case feedForwardUp = "mlp.c_fc"
    case feedForwardDown = "mlp.c_proj"
}

internal struct GPT2WeightSanitizerPlan: Equatable, Sendable {
    internal let layerCount: Int

    internal init(_ config: GPT2Configuration) {
        self.layerCount = config.hiddenLayers
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]

        for (checkpointKey, checkpointValue) in weights {
            let normalizedKey = Self.normalizedKey(checkpointKey)
            guard !shouldDrop(normalizedKey) else {
                continue
            }

            var value = checkpointValue
            if shouldTransposeRawProjection(normalizedKey) {
                value = checkpointValue.transposed(1, 0)
            }

            let modelKey = normalizedKey.hasPrefix("model.")
                ? normalizedKey
                : "model.\(normalizedKey)"
            sanitized[modelKey] = value
        }

        return sanitized
    }

    private static func normalizedKey(_ key: String) -> String {
        if key.hasPrefix("transformer.") {
            return String(key.dropFirst("transformer.".count))
        }
        return key
    }

    private func shouldDrop(_ key: String) -> Bool {
        let unprefixedKey = key.hasPrefix("model.") ? String(key.dropFirst("model.".count)) : key
        if unprefixedKey == "lm_head.weight" {
            return true
        }
        return unprefixedKey.hasPrefix("h.")
            && (unprefixedKey.hasSuffix(".attn.bias")
                || unprefixedKey.hasSuffix(".attn.masked_bias"))
    }

    private func shouldTransposeRawProjection(_ key: String) -> Bool {
        guard !key.hasPrefix("model.") else {
            return false
        }

        for layerIndex in 0 ..< layerCount {
            for projection in GPT2RawLinearProjection.allCases {
                if key == "h.\(layerIndex).\(projection.rawValue).weight" {
                    return true
                }
            }
        }
        return false
    }
}

// MARK: - Layers

internal final class GPT2Attention: Module {
    let layout: GPT2AttentionLayout

    @ModuleInfo(key: "c_attn") var combinedProjection: Linear
    @ModuleInfo(key: "c_proj") var outputProjection: Linear

    init(_ config: GPT2Configuration) {
        self.layout = GPT2AttentionLayout(config)

        _combinedProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.combinedProjectionSize,
            bias: true
        )
        _outputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let projected = combinedProjection(hiddenStates)
        let splitPoints = [layout.hiddenSize, 2 * layout.hiddenSize]
        let qkv = split(projected, indices: splitPoints, axis: -1)

        let queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

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

internal final class GPT2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var upProjection: Linear
    @ModuleInfo(key: "c_proj") var downProjection: Linear

    init(_ config: GPT2Configuration) {
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.feedForwardSize,
            bias: true
        )
        _downProjection.wrappedValue = Linear(
            config.feedForwardSize,
            config.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        downProjection(geluApproximate(upProjection(hiddenStates)))
    }
}

internal final class GPT2TransformerBlock: Module {
    @ModuleInfo(key: "attn") var attention: GPT2Attention
    @ModuleInfo(key: "mlp") var feedForward: GPT2FeedForward
    @ModuleInfo(key: "ln_1") var attentionLayerNorm: LayerNorm
    @ModuleInfo(key: "ln_2") var feedForwardLayerNorm: LayerNorm

    init(_ config: GPT2Configuration) {
        _attention.wrappedValue = GPT2Attention(config)
        _feedForward.wrappedValue = GPT2FeedForward(config)
        _attentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        _feedForwardLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let attended = hiddenStates + attention(
            attentionLayerNorm(hiddenStates),
            mask: mask,
            cache: cache
        )
        return attended + feedForward(feedForwardLayerNorm(attended))
    }
}

internal final class GPT2Backbone: Module {
    @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
    @ModuleInfo(key: "wpe") var positionEmbedding: Embedding
    @ModuleInfo(key: "h") var layers: [GPT2TransformerBlock]
    @ModuleInfo(key: "ln_f") var finalLayerNorm: LayerNorm

    init(_ config: GPT2Configuration) {
        precondition(config.vocabularySize > 0, "GPT-2 vocabulary size must be positive")
        precondition(
            config.maxPositionEmbeddings > 0,
            "GPT-2 position embedding count must be positive"
        )

        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.maxPositionEmbeddings,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            GPT2TransformerBlock(config)
        }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let tokenCount = inputs.dim(1)
        let cacheOffset = cache?.first?.offset ?? 0
        let positionIDs = (MLXArray(0 ..< tokenCount) + MLXArray(cacheOffset))[.newAxis, 0...]

        var hiddenStates = tokenEmbedding(inputs) + positionEmbedding(positionIDs)
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = createAttentionMask(
            h: hiddenStates,
            cache: cache
        )

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return finalLayerNorm(hiddenStates)
    }
}

internal final class GPT2Model: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let configuration: GPT2Configuration

    @ModuleInfo(key: "model") var model: GPT2Backbone

    init(_ config: GPT2Configuration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.attentionHeads, count: config.hiddenLayers)
        _model.wrappedValue = GPT2Backbone(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.tokenEmbedding.asLinear(model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: model.tokenEmbedding.asLinear(hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        GPT2WeightSanitizerPlan(configuration).sanitize(weights)
    }
}

extension GPT2Model: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["c_attn", "c_proj"]) }
    }
}
