import Foundation
import MLX
import MLXFast
import MLXNN

internal struct PhixtralAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let headSize: Int
    internal let rotaryDimensions: Int
    internal let attentionScale: Float

    internal init(_ config: PhixtralConfiguration) {
        precondition(config.hiddenSize > 0, "Phixtral hidden size must be positive")
        precondition(config.attentionHeads > 0, "Phixtral attention heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "Phixtral hidden size must be divisible by attention heads"
        )
        precondition(config.rotaryDimensions > 0, "Phixtral rotary dimensions must be positive")

        self.hiddenSize = config.hiddenSize
        self.attentionHeads = config.attentionHeads
        self.headSize = config.hiddenSize / config.attentionHeads
        self.rotaryDimensions = min(config.rotaryDimensions, headSize)
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }
}

internal struct PhixtralRouterPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let expertsPerToken: Int

    internal init(_ config: PhixtralConfiguration) {
        precondition(config.numLocalExperts > 0, "Phixtral expert count must be positive")
        precondition(config.numExpertsPerToken > 0, "Phixtral experts per token must be positive")
        precondition(
            config.numExpertsPerToken <= config.numLocalExperts,
            "Phixtral experts per token cannot exceed local experts"
        )

        self.expertCount = config.numLocalExperts
        self.expertsPerToken = config.numExpertsPerToken
    }
}

private final class PhixtralAttention: Module {
    private let layout: PhixtralAttentionLayout
    private let rope: RoPE

    @ModuleInfo(key: "Wqkv") private var queryKeyValueProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear

    init(_ config: PhixtralConfiguration) {
        let layout = PhixtralAttentionLayout(config)
        self.layout = layout
        self.rope = RoPE(
            dimensions: layout.rotaryDimensions,
            traditional: false
        )
        self._queryKeyValueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.hiddenSize * 3
        )
        self._outputProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.hiddenSize
        )
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = hiddenStates.dim(0)
        let tokenCount = hiddenStates.dim(1)
        let qkv = queryKeyValueProjection(hiddenStates).split(parts: 3, axis: -1)

        var queries = qkv[0]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = qkv[1]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = qkv[2]
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headSize)
            .transposed(0, 2, 1, 3)

        if let cache {
            queries = rope(queries, offset: cache.offset)
            keys = rope(keys, offset: cache.offset)
        } else {
            queries = rope(queries)
            keys = rope(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries.asType(.float32),
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        .asType(values.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, layout.hiddenSize)

        return outputProjection(output)
    }
}

private final class PhixtralSparseBlock: Module {
    private let routerPlan: PhixtralRouterPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var experts: SwitchMLP

    init(_ config: PhixtralConfiguration) {
        let routerPlan = PhixtralRouterPlan(config)
        self.routerPlan = routerPlan
        self._gate.wrappedValue = Linear(
            config.hiddenSize,
            routerPlan.expertCount,
            bias: false
        )
        self._experts.wrappedValue = SwitchMLP(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: routerPlan.expertCount,
            bias: true
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let logits = gate(hiddenStates)
        let expertIndices = stopGradient(
            argPartition(
                -logits,
                kth: routerPlan.expertsPerToken - 1,
                axis: -1
            )[.ellipsis, ..<routerPlan.expertsPerToken]
        )
        let routingWeights = softmax(
            takeAlong(logits, expertIndices, axis: -1),
            axis: -1,
            precise: true
        )

        return (experts(hiddenStates, expertIndices) * routingWeights[.ellipsis, .newAxis])
            .sum(axis: -2)
    }
}

private final class PhixtralBlock: Module {
    @ModuleInfo(key: "mixer") fileprivate var attention: PhixtralAttention
    @ModuleInfo(key: "ln") private var layerNorm: LayerNorm
    @ModuleInfo(key: "moe") private var sparseBlock: PhixtralSparseBlock

    init(_ config: PhixtralConfiguration) {
        self._attention.wrappedValue = PhixtralAttention(config)
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        self._sparseBlock.wrappedValue = PhixtralSparseBlock(config)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalizedStates = layerNorm(hiddenStates)
        return hiddenStates
            + attention(normalizedStates, mask: mask, cache: cache)
            + sparseBlock(normalizedStates)
    }
}

private final class PhixtralEmbedding: Module {
    @ModuleInfo(key: "wte") private var tokenEmbedding: Embedding

    init(_ config: PhixtralConfiguration) {
        self._tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        tokenEmbedding(tokens)
    }
}

private final class PhixtralTransformer: Module {
    @ModuleInfo(key: "embd") private var embeddings: PhixtralEmbedding
    @ModuleInfo(key: "h") fileprivate var layers: [PhixtralBlock]

    init(_ config: PhixtralConfiguration) {
        self._embeddings.wrappedValue = PhixtralEmbedding(config)
        self._layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            PhixtralBlock(config)
        }
    }

    func callAsFunction(
        _ tokens: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?
    ) -> MLXArray {
        var hiddenStates = embeddings(tokens)

        for (layerIndex, layer) in layers.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
        }

        return hiddenStates
    }
}

private final class PhixtralOutputHead: Module {
    @ModuleInfo(key: "ln") private var layerNorm: LayerNorm
    @ModuleInfo(key: "linear") private var linear: Linear

    init(_ config: PhixtralConfiguration) {
        self._layerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize,
            eps: config.layerNormEps
        )
        self._linear.wrappedValue = Linear(
            config.hiddenSize,
            config.vocabularySize
        )
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        linear(layerNorm(hiddenStates))
    }
}

internal final class PhixtralModel: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "transformer") fileprivate var transformer: PhixtralTransformer
    @ModuleInfo(key: "lm_head") private var lmHead: PhixtralOutputHead

    private let config: PhixtralConfiguration

    public init(_ config: PhixtralConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.attentionHeads, count: config.hiddenLayers)
        self._transformer.wrappedValue = PhixtralTransformer(config)
        self._lmHead.wrappedValue = PhixtralOutputHead(config)
    }

    public func callAsFunction(_ tokens: MLXArray, cache: [KVCache]?) -> MLXArray {
        let mask = createAttentionMask(h: tokens, cache: cache)
        return lmHead(transformer(tokens, mask: mask, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let tokens = input[text: .newAxis].tokens
        let mask = createAttentionMask(h: tokens, cache: cache)
        let hiddenStates = transformer(tokens, mask: mask, cache: cache)
        return greedyTokenOutput(logits: lmHead(lastTokenHiddenState(hiddenStates)), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for layerIndex in 0 ..< config.hiddenLayers {
            let layerPrefix = "transformer.h.\(layerIndex).moe"
            Self.packExpertWeights(
                in: &sanitized,
                sourcePrefix: "\(layerPrefix).mlp",
                targetPrefix: "\(layerPrefix).switch_mlp",
                expertCount: config.numLocalExperts
            )
        }

        return sanitized
    }

    private static func packExpertWeights(
        in weights: inout [String: MLXArray],
        sourcePrefix: String,
        targetPrefix: String,
        expertCount: Int
    ) {
        for projection in ["fc1", "fc2"] {
            for tensorName in ["weight", "scales", "biases", "bias"] {
                let firstKey = "\(sourcePrefix).0.\(projection).\(tensorName)"
                guard weights[firstKey] != nil else {
                    continue
                }

                var tensors: [MLXArray] = []
                tensors.reserveCapacity(expertCount)
                for expertIndex in 0 ..< expertCount {
                    let key = "\(sourcePrefix).\(expertIndex).\(projection).\(tensorName)"
                    guard let tensor = weights[key] else {
                        tensors.removeAll(keepingCapacity: true)
                        break
                    }
                    tensors.append(tensor)
                }
                guard tensors.count == expertCount else {
                    continue
                }

                for expertIndex in 0 ..< expertCount {
                    weights["\(sourcePrefix).\(expertIndex).\(projection).\(tensorName)"] = nil
                }
                weights["\(targetPrefix).\(projection).\(tensorName)"] = MLX.stacked(tensors)
            }
        }
    }
}

internal struct PhixtralConfiguration: Codable, Sendable, Equatable {
    internal var modelType: String
    internal var vocabularySize: Int
    internal var hiddenSize: Int
    internal var attentionHeads: Int
    internal var hiddenLayers: Int
    internal var rotaryDimensions: Int
    internal var intermediateSize: Int
    internal var numExpertsPerToken: Int
    internal var numLocalExperts: Int
    internal var layerNormEps: Float

    internal init(
        modelType: String = "phixtral",
        vocabularySize: Int = 51_200,
        hiddenSize: Int = 2_560,
        attentionHeads: Int = 32,
        hiddenLayers: Int = 32,
        rotaryDimensions: Int = 32,
        intermediateSize: Int? = nil,
        numExpertsPerToken: Int = 2,
        numLocalExperts: Int = 4,
        layerNormEps: Float = 1e-5
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.rotaryDimensions = rotaryDimensions
        self.intermediateSize = intermediateSize ?? hiddenSize * 4
        self.numExpertsPerToken = numExpertsPerToken
        self.numLocalExperts = numLocalExperts
        self.layerNormEps = layerNormEps
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case numVocab = "num_vocab"
        case hiddenSize = "hidden_size"
        case modelDim = "model_dim"
        case nEmbd = "n_embd"
        case attentionHeads = "num_attention_heads"
        case numHeads = "num_heads"
        case nHead = "n_head"
        case hiddenLayers = "num_hidden_layers"
        case numLayers = "num_layers"
        case nLayer = "n_layer"
        case rotaryDimensions = "rotary_dim"
        case intermediateSize = "intermediate_size"
        case nInner = "n_inner"
        case numExpertsPerToken = "num_experts_per_tok"
        case numLocalExperts = "num_local_experts"
        case layerNormEps = "layer_norm_epsilon"
        case layerNormEpsAlternate = "layer_norm_eps"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenSize = try Self.decodeInt(
            from: container,
            keys: [.hiddenSize, .modelDim, .nEmbd],
            default: 2_560
        )

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "phixtral",
            vocabularySize: try Self.decodeInt(
                from: container,
                keys: [.vocabularySize, .numVocab],
                default: 51_200
            ),
            hiddenSize: hiddenSize,
            attentionHeads: try Self.decodeInt(
                from: container,
                keys: [.attentionHeads, .numHeads, .nHead],
                default: 32
            ),
            hiddenLayers: try Self.decodeInt(
                from: container,
                keys: [.hiddenLayers, .numLayers, .nLayer],
                default: 32
            ),
            rotaryDimensions: try container.decodeIfPresent(
                Int.self,
                forKey: .rotaryDimensions
            ) ?? 32,
            intermediateSize: try Self.decodeOptionalInt(
                from: container,
                keys: [.intermediateSize, .nInner]
            ),
            numExpertsPerToken: try container.decodeIfPresent(
                Int.self,
                forKey: .numExpertsPerToken
            ) ?? 2,
            numLocalExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .numLocalExperts
            ) ?? 4,
            layerNormEps: try Self.decodeFloat(
                from: container,
                keys: [.layerNormEps, .layerNormEpsAlternate],
                default: 1e-5
            )
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(hiddenSize, forKey: .modelDim)
        try container.encode(attentionHeads, forKey: .numHeads)
        try container.encode(hiddenLayers, forKey: .numLayers)
        try container.encode(rotaryDimensions, forKey: .rotaryDimensions)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try container.encode(numLocalExperts, forKey: .numLocalExperts)
        try container.encode(layerNormEps, forKey: .layerNormEps)
    }

    private static func decodeInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys],
        default defaultValue: Int
    ) throws -> Int {
        if let value = try decodeOptionalInt(from: container, keys: keys) {
            return value
        }
        return defaultValue
    }

    private static func decodeOptionalInt(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys]
    ) throws -> Int? {
        for key in keys {
            if let value = try container.decodeIfPresent(Int.self, forKey: key) {
                return value
            }
        }
        return nil
    }

    private static func decodeFloat(
        from container: KeyedDecodingContainer<CodingKeys>,
        keys: [CodingKeys],
        default defaultValue: Float
    ) throws -> Float {
        for key in keys {
            if let value = try container.decodeIfPresent(Float.self, forKey: key) {
                return value
            }
        }
        return defaultValue
    }
}

extension PhixtralModel: LoRAModel {
    func loraLinearLayers() -> LoRALinearLayers {
        transformer.layers.map { ($0.attention, ["Wqkv", "out_proj"]) }
    }
}
