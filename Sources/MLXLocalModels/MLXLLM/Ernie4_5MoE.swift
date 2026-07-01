import Foundation
import MLX
import MLXNN

internal struct Ernie45MoEAttentionLayout: Equatable, Sendable {
    internal let hiddenSize: Int
    internal let queryHeads: Int
    internal let keyValueHeads: Int
    internal let headSize: Int
    internal let queryProjectionSize: Int
    internal let keyValueProjectionSize: Int
    internal let attentionScale: Float

    internal init(_ config: Ernie45MoEConfiguration) {
        precondition(config.hiddenSize > 0, "hidden_size must be positive")
        precondition(config.numAttentionHeads > 0, "num_attention_heads must be positive")
        precondition(config.numKeyValueHeads > 0, "num_key_value_heads must be positive")

        let headSize = Self.resolveHeadSize(config)
        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.numAttentionHeads
        self.keyValueHeads = config.numKeyValueHeads
        self.headSize = headSize
        self.queryProjectionSize = queryHeads * headSize
        self.keyValueProjectionSize = keyValueHeads * headSize
        self.attentionScale = 1 / Float(headSize).squareRoot()
    }

    private static func resolveHeadSize(_ config: Ernie45MoEConfiguration) -> Int {
        if let headDim = config.headDim {
            precondition(headDim > 0, "head_dim must be positive")
            return headDim
        }

        precondition(
            config.hiddenSize.isMultiple(of: config.numAttentionHeads),
            "hidden_size must be divisible by num_attention_heads when head_dim is absent"
        )
        return config.hiddenSize / config.numAttentionHeads
    }
}

internal enum Ernie45MoEGateActivation: String, Codable, Equatable, Sendable {
    case sigmoid
    case softmax

    internal func apply(to logits: MLXArray) -> MLXArray {
        switch self {
        case .sigmoid:
            MLX.sigmoid(logits.asType(.float32))
        case .softmax:
            MLX.softmax(logits.asType(.float32), axis: -1, precise: true)
        }
    }
}

internal struct Ernie45MoERoutingPlan: Equatable, Sendable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let gateActivation: Ernie45MoEGateActivation

    internal init(_ config: Ernie45MoEConfiguration) {
        precondition(config.moeNumExperts > 0, "moe_num_experts must be positive")
        precondition(config.moeK > 0, "moe_k must be positive")
        precondition(
            config.moeK <= config.moeNumExperts,
            "moe_k cannot exceed moe_num_experts"
        )

        self.expertCount = config.moeNumExperts
        self.selectedExpertCount = config.moeK
        self.gateActivation = config.moeGateActivation
    }

    internal func route(_ logits: MLXArray) -> (scores: MLXArray, indices: MLXArray) {
        let gates = gateActivation.apply(to: logits)
        let indices = stopGradient(
            argPartition(-gates, kth: selectedExpertCount - 1, axis: -1)[
                .ellipsis,
                ..<selectedExpertCount
            ]
        )
        var scores = takeAlong(gates, indices, axis: -1)
        scores = scores / maximum(scores.sum(axis: -1, keepDims: true), MLXArray(1e-12))
        return (scores, indices)
    }
}

internal struct Ernie45MoELayerPlan: Equatable, Sendable {
    internal let startIndex: Int
    internal let endIndex: Int
    internal let interval: Int

    internal init(_ config: Ernie45MoEConfiguration) {
        precondition(config.numHiddenLayers > 0, "num_hidden_layers must be positive")
        precondition(config.moeLayerInterval > 0, "moe_layer_interval must be positive")
        precondition(!config.moeLayerStartIndex.values.isEmpty, "moe_layer_start_index is empty")

        self.startIndex = config.moeLayerStartIndex.values.min() ?? 0
        self.endIndex = config.moeLayerEndIndex?.values.max() ?? config.numHiddenLayers - 1
        self.interval = config.moeLayerInterval
    }

    internal func usesSparseExperts(layerIndex: Int) -> Bool {
        ((layerIndex + 1) % interval == 0)
            && layerIndex >= startIndex
            && layerIndex <= endIndex
    }
}

private enum Ernie45MoEExpertProjection: String, CaseIterable {
    case gate = "gate_proj"
    case down = "down_proj"
    case up = "up_proj"
}

internal struct Ernie45MoEExpertPackingPlan: Equatable, Sendable {
    internal let layerCount: Int
    internal let expertCount: Int

    internal init(_ config: Ernie45MoEConfiguration) {
        self.layerCount = config.numHiddenLayers
        self.expertCount = config.moeNumExperts
    }

    internal func pack(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var packed = weights

        for layerIndex in 0 ..< layerCount {
            let prefix = "model.layers.\(layerIndex).mlp"
            for projection in Ernie45MoEExpertProjection.allCases {
                for tensorName in ["weight", "bias", "scales", "biases"] {
                    pack(
                        projection.rawValue,
                        tensorName: tensorName,
                        prefix: prefix,
                        weights: &packed
                    )
                }
            }
        }

        return packed
    }

    private func pack(
        _ projection: String,
        tensorName: String,
        prefix: String,
        weights: inout [String: MLXArray]
    ) {
        let keys = (0 ..< expertCount).map { expertIndex in
            "\(prefix).experts.\(expertIndex).\(projection).\(tensorName)"
        }
        let tensors = keys.compactMap { weights[$0] }
        guard tensors.count == expertCount else {
            return
        }

        for key in keys {
            weights[key] = nil
        }
        weights["\(prefix).switch_mlp.\(projection).\(tensorName)"] = stacked(tensors)
    }
}

private struct Ernie45MoEWeightSanitizer {
    let config: Ernie45MoEConfiguration

    func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        sanitized = sanitized.filter { key, _ in
            !Self.unusedCheckpointPatterns.contains { key.contains($0) }
        }

        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }

        return Ernie45MoEExpertPackingPlan(config).pack(sanitized)
    }

    private static let unusedCheckpointPatterns = [
        "mtp_block.",
        "mtp_linear_proj.",
        "mtp_hidden_norm.",
        "mtp_emb_norm.",
        "e_score_correction_bias",
        "rotary_emb.inv_freq"
    ]
}

private final class Ernie45MoEAttention: Module {
    private let layout: Ernie45MoEAttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: Ernie45MoEConfiguration) {
        let layout = Ernie45MoEAttentionLayout(config)
        self.layout = layout
        self.rope = initializeRope(
            dims: layout.headSize,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: nil,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )

        self._queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.useBias
        )
        self._keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useBias
        )
        self._valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.useBias
        )
        self._outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.useBias
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
            .reshaped(batchSize, tokenCount, layout.queryHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        var keys = keyProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
            .transposed(0, 2, 1, 3)
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headSize)
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

private final class Ernie45MoEDenseFeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(hiddenSize: Int, intermediateSize: Int, useBias: Bool) {
        self._gateProjection.wrappedValue = Linear(
            hiddenSize,
            intermediateSize,
            bias: useBias
        )
        self._downProjection.wrappedValue = Linear(
            intermediateSize,
            hiddenSize,
            bias: useBias
        )
        self._upProjection.wrappedValue = Linear(
            hiddenSize,
            intermediateSize,
            bias: useBias
        )
    }

    convenience init(_ config: Ernie45MoEConfiguration) {
        self.init(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            useBias: config.useBias
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class Ernie45MoESparseFeedForward: Module, UnaryLayer {
    private let routingPlan: Ernie45MoERoutingPlan

    @ModuleInfo(key: "gate") private var gate: Linear
    @ModuleInfo(key: "switch_mlp") private var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") private var sharedExperts: Ernie45MoEDenseFeedForward?

    init(_ config: Ernie45MoEConfiguration) {
        self.routingPlan = Ernie45MoERoutingPlan(config)
        let routedIntermediateSize = config.resolvedMoEIntermediateSize

        self._gate.wrappedValue = Linear(
            config.hiddenSize,
            routingPlan.expertCount,
            bias: false
        )
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: routedIntermediateSize,
            numExperts: routingPlan.expertCount,
            bias: config.useBias
        )

        if config.moeNumSharedExperts > 0 {
            self._sharedExperts.wrappedValue = Ernie45MoEDenseFeedForward(
                hiddenSize: config.hiddenSize,
                intermediateSize: routedIntermediateSize * config.moeNumSharedExperts,
                useBias: config.useBias
            )
        }
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let route = routingPlan.route(gate(input))
        let expertOutput = switchMLP(input, route.indices)
        var output = (expertOutput * route.scores[.ellipsis, .newAxis])
            .sum(axis: -2)
            .asType(expertOutput.dtype)

        if let sharedExperts {
            output = output + sharedExperts(input)
        }

        return output
    }
}

private final class Ernie45MoEBlock: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: Ernie45MoEAttention
    @ModuleInfo(key: "mlp") private var feedForward: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: Ernie45MoEConfiguration, layerIndex: Int, layerPlan: Ernie45MoELayerPlan) {
        self._attention.wrappedValue = Ernie45MoEAttention(config)
        if layerPlan.usesSparseExperts(layerIndex: layerIndex) {
            self._feedForward.wrappedValue = Ernie45MoESparseFeedForward(config)
        } else {
            self._feedForward.wrappedValue = Ernie45MoEDenseFeedForward(config)
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
        let attentionOutput = attention(
            inputLayerNorm(input),
            mask: mask,
            cache: cache
        )
        let afterAttention = input + attentionOutput
        return afterAttention + feedForward(postAttentionLayerNorm(afterAttention))
    }
}

private final class Ernie45MoEBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embeddings: Embedding
    @ModuleInfo fileprivate var layers: [Ernie45MoEBlock]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: Ernie45MoEConfiguration) {
        precondition(config.vocabularySize > 0, "vocab_size must be positive")

        let layerPlan = Ernie45MoELayerPlan(config)
        self._embeddings.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        self.layers = (0 ..< config.numHiddenLayers).map { layerIndex in
            Ernie45MoEBlock(config, layerIndex: layerIndex, layerPlan: layerPlan)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ tokens: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hiddenStates = embeddings(tokens)
        let mask = createAttentionMask(h: hiddenStates, cache: cache)

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: mask,
                cache: cache?[index]
            )
        }

        return norm(hiddenStates)
    }
}

internal final class Ernie45MoEModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    private let config: Ernie45MoEConfiguration
    @ModuleInfo(key: "model") private var model: Ernie45MoEBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    public init(_ config: Ernie45MoEConfiguration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(
            repeating: config.numKeyValueHeads,
            count: config.numHiddenLayers
        )
        self._model.wrappedValue = Ernie45MoEBackbone(config)

        if !config.tieWordEmbeddings {
            self._lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
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
                from: lastTokenHiddenState(
                    model(input[text: .newAxis].tokens, cache: cache)
                )
            ),
            state: state
        )
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Ernie45MoEWeightSanitizer(config: config).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        lmHead.map { $0(hiddenStates) }
            ?? model.embeddings.asLinear(hiddenStates)
    }
}

internal struct Ernie45MoEConfiguration: Codable, Sendable, Equatable {
    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int?
    var numHiddenLayers: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var ropeTheta: Float
    var useBias: Bool
    var tieWordEmbeddings: Bool
    var moeNumExperts: Int
    var moeLayerStartIndex: IntOrIntArray
    var moeIntermediateSize: Int
    var moeCapacity: [Int]
    var moeK: Int
    var moeLayerInterval: Int
    var moeUseAuxFree: Bool
    var moeNumSharedExperts: Int
    var moeLayerEndIndex: IntOrIntArray?
    var moeGateActivation: Ernie45MoEGateActivation

    internal var resolvedMoEIntermediateSize: Int {
        moeIntermediateSize > 0 ? moeIntermediateSize : intermediateSize
    }

    internal init(
        modelType: String = "ernie4_5_moe",
        hiddenSize: Int = 2_560,
        intermediateSize: Int = 12_288,
        maxPositionEmbeddings: Int = 131_072,
        numAttentionHeads: Int = 20,
        numKeyValueHeads: Int = 4,
        headDim: Int? = nil,
        numHiddenLayers: Int = 28,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int = 103_424,
        ropeTheta: Float = 500_000,
        useBias: Bool = false,
        tieWordEmbeddings: Bool = true,
        moeNumExperts: Int = 64,
        moeLayerStartIndex: IntOrIntArray = IntOrIntArray([0]),
        moeIntermediateSize: Int = 0,
        moeCapacity: [Int] = [],
        moeK: Int = 1,
        moeLayerInterval: Int = 1,
        moeUseAuxFree: Bool = false,
        moeNumSharedExperts: Int = 0,
        moeLayerEndIndex: IntOrIntArray? = nil,
        moeGateActivation: Ernie45MoEGateActivation = .softmax
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numHiddenLayers = numHiddenLayers
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.ropeTheta = ropeTheta
        self.useBias = useBias
        self.tieWordEmbeddings = tieWordEmbeddings
        self.moeNumExperts = moeNumExperts
        self.moeLayerStartIndex = moeLayerStartIndex
        self.moeIntermediateSize = moeIntermediateSize
        self.moeCapacity = moeCapacity
        self.moeK = moeK
        self.moeLayerInterval = moeLayerInterval
        self.moeUseAuxFree = moeUseAuxFree
        self.moeNumSharedExperts = moeNumSharedExperts
        self.moeLayerEndIndex = moeLayerEndIndex
        self.moeGateActivation = moeGateActivation
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numHiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case useBias = "use_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case moeNumExperts = "moe_num_experts"
        case moeLayerStartIndex = "moe_layer_start_index"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeCapacity = "moe_capacity"
        case moeK = "moe_k"
        case moeLayerInterval = "moe_layer_interval"
        case moeUseAuxFree = "moe_use_aux_free"
        case moeNumSharedExperts = "moe_num_shared_experts"
        case moeLayerEndIndex = "moe_layer_end_index"
        case moeGateActivation = "moe_gate_act"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: Self.CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "ernie4_5_moe",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? 2_560,
            intermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .intermediateSize
            ) ?? 12_288,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            numAttentionHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .numAttentionHeads
            ) ?? 20,
            numKeyValueHeads: try container.decodeIfPresent(
                Int.self,
                forKey: .numKeyValueHeads
            ) ?? 4,
            headDim: try container.decodeIfPresent(Int.self, forKey: .headDim),
            numHiddenLayers: try container.decodeIfPresent(
                Int.self,
                forKey: .numHiddenLayers
            ) ?? 28,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(
                Int.self,
                forKey: .vocabularySize
            ) ?? 103_424,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            useBias: try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            moeNumExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .moeNumExperts
            ) ?? 64,
            moeLayerStartIndex: try container.decodeIfPresent(
                IntOrIntArray.self,
                forKey: .moeLayerStartIndex
            ) ?? IntOrIntArray([0]),
            moeIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .moeIntermediateSize
            ) ?? 0,
            moeCapacity: try container.decodeIfPresent([Int].self, forKey: .moeCapacity)
                ?? [],
            moeK: try container.decodeIfPresent(Int.self, forKey: .moeK) ?? 1,
            moeLayerInterval: try container.decodeIfPresent(
                Int.self,
                forKey: .moeLayerInterval
            ) ?? 1,
            moeUseAuxFree: try container.decodeIfPresent(
                Bool.self,
                forKey: .moeUseAuxFree
            ) ?? false,
            moeNumSharedExperts: try container.decodeIfPresent(
                Int.self,
                forKey: .moeNumSharedExperts
            ) ?? 0,
            moeLayerEndIndex: try container.decodeIfPresent(
                IntOrIntArray.self,
                forKey: .moeLayerEndIndex
            ),
            moeGateActivation: try container.decodeIfPresent(
                Ernie45MoEGateActivation.self,
                forKey: .moeGateActivation
            ) ?? .softmax
        )
    }
}

extension Ernie45MoEModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
