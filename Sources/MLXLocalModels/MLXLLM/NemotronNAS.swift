import Foundation
import MLX
import MLXFast
import MLXNN

internal struct NemotronNASAttentionConfig: Codable, Equatable, Sendable {
    var noOp: Bool
    var replaceWithLinear: Bool
    var sparsify: [String]?
    var headsPerKVGroup: Int?
    var windowLength: Int?
    var sinkTokenCount: Int?
    var usePrefillWindowInSinkAttention: Bool
    var unshiftedSink: Bool

    internal init(
        noOp: Bool = false,
        replaceWithLinear: Bool = false,
        sparsify: [String]? = nil,
        headsPerKVGroup: Int? = nil,
        windowLength: Int? = nil,
        sinkTokenCount: Int? = nil,
        usePrefillWindowInSinkAttention: Bool = false,
        unshiftedSink: Bool = false
    ) {
        self.noOp = noOp
        self.replaceWithLinear = replaceWithLinear
        self.sparsify = sparsify
        self.headsPerKVGroup = headsPerKVGroup
        self.windowLength = windowLength
        self.sinkTokenCount = sinkTokenCount
        self.usePrefillWindowInSinkAttention = usePrefillWindowInSinkAttention
        self.unshiftedSink = unshiftedSink
    }

    var usesRealAttention: Bool {
        !noOp && !replaceWithLinear
    }

    var usesAnySubblock: Bool {
        !noOp
    }

    private enum CodingKeys: String, CodingKey {
        case noOp = "no_op"
        case replaceWithLinear = "replace_with_linear"
        case sparsify
        case headsPerKVGroup = "n_heads_in_group"
        case windowLength = "window_length"
        case sinkTokenCount = "num_sink_tokens"
        case usePrefillWindowInSinkAttention = "use_prefill_window_in_sink_attention"
        case unshiftedSink = "unshifted_sink"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            noOp: try container.decodeIfPresent(Bool.self, forKey: .noOp) ?? false,
            replaceWithLinear: try container.decodeIfPresent(
                Bool.self,
                forKey: .replaceWithLinear
            ) ?? false,
            sparsify: try container.decodeIfPresent([String].self, forKey: .sparsify),
            headsPerKVGroup: try container.decodeIfPresent(
                Int.self,
                forKey: .headsPerKVGroup
            ),
            windowLength: try container.decodeIfPresent(Int.self, forKey: .windowLength),
            sinkTokenCount: try container.decodeIfPresent(Int.self, forKey: .sinkTokenCount),
            usePrefillWindowInSinkAttention: try container.decodeIfPresent(
                Bool.self,
                forKey: .usePrefillWindowInSinkAttention
            ) ?? false,
            unshiftedSink: try container.decodeIfPresent(Bool.self, forKey: .unshiftedSink)
                ?? false
        )
    }
}

internal struct NemotronNASFeedForwardConfig: Codable, Equatable, Sendable {
    var noOp: Bool
    var replaceWithLinear: Bool
    var sparsify: [String]?
    var multiplier: Float?

    internal init(
        noOp: Bool = false,
        replaceWithLinear: Bool = false,
        sparsify: [String]? = nil,
        multiplier: Float? = nil
    ) {
        self.noOp = noOp
        self.replaceWithLinear = replaceWithLinear
        self.sparsify = sparsify
        self.multiplier = multiplier
    }

    var usesAnySubblock: Bool {
        !noOp
    }

    private enum CodingKeys: String, CodingKey {
        case noOp = "no_op"
        case replaceWithLinear = "replace_with_linear"
        case sparsify
        case multiplier = "ffn_mult"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            noOp: try container.decodeIfPresent(Bool.self, forKey: .noOp) ?? false,
            replaceWithLinear: try container.decodeIfPresent(
                Bool.self,
                forKey: .replaceWithLinear
            ) ?? false,
            sparsify: try container.decodeIfPresent([String].self, forKey: .sparsify),
            multiplier: try container.decodeIfPresent(Float.self, forKey: .multiplier)
        )
    }
}

internal struct NemotronNASBlockConfig: Codable, Equatable, Sendable {
    var attention: NemotronNASAttentionConfig
    var feedForward: NemotronNASFeedForwardConfig

    internal init(
        attention: NemotronNASAttentionConfig = .init(),
        feedForward: NemotronNASFeedForwardConfig = .init()
    ) {
        self.attention = attention
        self.feedForward = feedForward
    }

    private enum CodingKeys: String, CodingKey {
        case attention
        case feedForward = "ffn"
    }
}

internal struct NemotronNASConfiguration: Codable, Equatable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var blockConfigs: [NemotronNASBlockConfig]
    var hiddenActivation: String
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "nemotron-nas",
        hiddenSize: Int,
        hiddenLayers: Int,
        attentionHeads: Int,
        rmsNormEps: Float = 1e-5,
        vocabularySize: Int,
        blockConfigs: [NemotronNASBlockConfig],
        hiddenActivation: String = "silu",
        attentionBias: Bool = false,
        mlpBias: Bool = false,
        ropeTheta: Float = 500_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        maxPositionEmbeddings: Int = 131_072,
        tieWordEmbeddings: Bool = false
    ) {
        precondition(
            blockConfigs.count == hiddenLayers,
            "Nemotron-NAS block config count must match hidden layer count"
        )
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.blockConfigs = blockConfigs
        self.hiddenActivation = hiddenActivation
        self.attentionBias = attentionBias
        self.mlpBias = mlpBias
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    var realAttentionLayerCount: Int {
        blockConfigs.filter(\.attention.usesRealAttention).count
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case blockConfigs = "block_configs"
        case hiddenActivation = "hidden_act"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 80
        let blockConfigs = try container.decode(
            [NemotronNASBlockConfig].self,
            forKey: .blockConfigs
        )
        guard blockConfigs.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .blockConfigs,
                in: container,
                debugDescription: "block_configs count must match num_hidden_layers"
            )
        }

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "nemotron-nas",
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 8_192,
            hiddenLayers: hiddenLayers,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? 64,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5,
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 128_256,
            blockConfigs: blockConfigs,
            hiddenActivation: try container.decodeIfPresent(
                String.self,
                forKey: .hiddenActivation
            ) ?? "silu",
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 500_000,
            ropeScaling: try container.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling
            ),
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? false
        )
    }
}

internal struct NemotronNASAttentionLayout: Equatable, Sendable {
    let hiddenSize: Int
    let queryHeads: Int
    let keyValueHeads: Int
    let headDim: Int
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let attentionScale: Float

    init(_ config: NemotronNASConfiguration, attention: NemotronNASAttentionConfig) {
        precondition(config.hiddenSize > 0, "Nemotron-NAS hidden size must be positive")
        precondition(config.attentionHeads > 0, "Nemotron-NAS attention heads must be positive")
        precondition(
            config.hiddenSize.isMultiple(of: config.attentionHeads),
            "Nemotron-NAS hidden size must divide evenly across attention heads"
        )
        let headsPerKVGroup = attention.headsPerKVGroup ?? 1
        precondition(headsPerKVGroup > 0, "Nemotron-NAS GQA group size must be positive")
        precondition(
            config.attentionHeads.isMultiple(of: headsPerKVGroup),
            "Nemotron-NAS attention heads must divide into GQA groups"
        )

        self.hiddenSize = config.hiddenSize
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.attentionHeads / headsPerKVGroup
        self.headDim = config.hiddenSize / config.attentionHeads
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.attentionScale = pow(Float(headDim), -0.5)
    }
}

private enum NemotronNASActivation: String {
    case siluActivation = "silu"
    case reluActivation = "relu"
    case geluActivation = "gelu"
    case geluNew = "gelu_new"
    case geluFast = "gelu_fast"

    init(name: String) {
        self = Self(rawValue: name) ?? .siluActivation
    }

    func apply(to input: MLXArray) -> MLXArray {
        switch self {
        case .siluActivation:
            return silu(input)
        case .reluActivation:
            return relu(input)
        case .geluActivation:
            return gelu(input)
        case .geluNew, .geluFast:
            return geluApproximate(input)
        }
    }
}

private func nemotronNASFeedForwardSize(multiplier: Float, hiddenSize: Int) -> Int {
    let unrounded = Int(2 * multiplier * Float(hiddenSize) / 3)
    let multiple = 256
    let remainder = unrounded % multiple
    return remainder == 0 ? unrounded : unrounded + multiple - remainder
}

private final class NemotronNASLinearReplacement: Module, UnaryLayer {
    @ModuleInfo(key: "linear") private var linear: Linear

    init(hiddenSize: Int, bias: Bool) {
        _linear.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        linear(input)
    }
}

private final class NemotronNASAttention: Module {
    private let layout: NemotronNASAttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: NemotronNASConfiguration, attention: NemotronNASAttentionConfig) {
        self.layout = NemotronNASAttentionLayout(config, attention: attention)
        self.rope = initializeRope(
            dims: layout.headDim,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings
        )
        _queryProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryProjectionSize,
            bias: config.attentionBias
        )
        _keyProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _valueProjection.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueProjectionSize,
            bias: config.attentionBias
        )
        _outputProjection.wrappedValue = Linear(
            layout.queryProjectionSize,
            layout.hiddenSize,
            bias: config.attentionBias
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        let offset = cache?.offset ?? 0

        let queries = rope(
            queryProjection(input)
                .reshaped(batchSize, sequenceLength, layout.queryHeads, layout.headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = rope(
            keyProjection(input)
                .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(input)
            .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
            .transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        return outputProjection(
            output
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, -1)
        )
    }
}

private final class NemotronNASMLP: Module, UnaryLayer {
    private let activation: NemotronNASActivation

    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: NemotronNASConfiguration, feedForward: NemotronNASFeedForwardConfig) {
        self.activation = NemotronNASActivation(name: config.hiddenActivation)
        let hiddenSize = nemotronNASFeedForwardSize(
            multiplier: feedForward.multiplier ?? 1,
            hiddenSize: config.hiddenSize
        )
        _gateProjection.wrappedValue = Linear(
            config.hiddenSize,
            hiddenSize,
            bias: config.mlpBias
        )
        _downProjection.wrappedValue = Linear(
            hiddenSize,
            config.hiddenSize,
            bias: config.mlpBias
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenSize,
            hiddenSize,
            bias: config.mlpBias
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(activation.apply(to: gateProjection(input)) * upProjection(input))
    }
}

private enum NemotronNASSubblockKind: Equatable {
    case none
    case linear
    case attention
    case feedForward
}

private final class NemotronNASDecoderLayer: Module {
    let attentionKind: NemotronNASSubblockKind
    let feedForwardKind: NemotronNASSubblockKind

    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm?
    @ModuleInfo(key: "self_attn") fileprivate var attentionBlock: Module?
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm?
    @ModuleInfo(key: "mlp") fileprivate var feedForwardBlock: Module?

    init(_ config: NemotronNASConfiguration, layerIndex: Int) {
        let block = config.blockConfigs[layerIndex]

        if block.attention.noOp {
            self.attentionKind = .none
        } else {
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            if block.attention.replaceWithLinear {
                self.attentionKind = .linear
                _attentionBlock.wrappedValue = NemotronNASLinearReplacement(
                    hiddenSize: config.hiddenSize,
                    bias: config.attentionBias
                )
            } else {
                self.attentionKind = .attention
                _attentionBlock.wrappedValue = NemotronNASAttention(
                    config,
                    attention: block.attention
                )
            }
        }

        if block.feedForward.noOp {
            self.feedForwardKind = .none
        } else {
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps
            )
            if block.feedForward.replaceWithLinear {
                self.feedForwardKind = .linear
                _feedForwardBlock.wrappedValue = NemotronNASLinearReplacement(
                    hiddenSize: config.hiddenSize,
                    bias: config.mlpBias
                )
            } else {
                self.feedForwardKind = .feedForward
                _feedForwardBlock.wrappedValue = NemotronNASMLP(
                    config,
                    feedForward: block.feedForward
                )
            }
        }
    }

    var usesRealAttention: Bool {
        attentionKind == .attention
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hiddenStates = input

        if attentionKind != .none {
            guard let inputLayerNorm else {
                preconditionFailure("Nemotron-NAS attention block is missing input norm")
            }
            let normalized = inputLayerNorm(hiddenStates)
            let attended: MLXArray
            switch attentionKind {
            case .attention:
                guard let attention = attentionBlock as? NemotronNASAttention else {
                    preconditionFailure("Nemotron-NAS attention block is missing attention")
                }
                attended = attention(normalized, mask: mask, cache: cache)
            case .linear:
                guard let replacement = attentionBlock as? NemotronNASLinearReplacement else {
                    preconditionFailure("Nemotron-NAS attention block is missing linear replacement")
                }
                attended = replacement(normalized)
            case .none, .feedForward:
                preconditionFailure("Invalid Nemotron-NAS attention block kind")
            }
            hiddenStates = hiddenStates + attended
        }

        if feedForwardKind != .none {
            guard let postAttentionLayerNorm else {
                preconditionFailure("Nemotron-NAS feed-forward block is missing norm")
            }
            let normalized = postAttentionLayerNorm(hiddenStates)
            let output: MLXArray
            switch feedForwardKind {
            case .feedForward, .linear:
                guard let feedForward = feedForwardBlock as? any UnaryLayer else {
                    preconditionFailure("Nemotron-NAS feed-forward block is missing module")
                }
                output = feedForward(normalized)
            case .none, .attention:
                preconditionFailure("Invalid Nemotron-NAS feed-forward block kind")
            }
            hiddenStates = hiddenStates + output
        }

        return hiddenStates
    }
}

private final class NemotronNASBackbone: Module {
    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [NemotronNASDecoderLayer]
    @ModuleInfo(key: "norm") private var norm: RMSNorm

    init(_ config: NemotronNASConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            NemotronNASDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        let firstCache = cache?.first
        let mask = createAttentionMask(h: hiddenStates, cache: firstCache)

        var cacheIndex = 0
        for layer in layers {
            let layerCache: KVCache?
            if layer.usesRealAttention {
                layerCache = cache?[cacheIndex]
                cacheIndex += 1
            } else {
                layerCache = nil
            }
            hiddenStates = layer(hiddenStates, mask: mask, cache: layerCache)
        }

        return norm(hiddenStates)
    }
}

internal final class NemotronNASModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel
{
    internal let modelType: String
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: NemotronNASConfiguration
    private let model: NemotronNASBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ configuration: NemotronNASConfiguration) {
        self.configuration = configuration
        self.modelType = configuration.modelType
        self.vocabularySize = configuration.vocabularySize
        self.model = NemotronNASBackbone(configuration)
        self.kvHeads = configuration.blockConfigs.compactMap { block in
            guard block.attention.usesRealAttention else {
                return nil
            }
            return configuration.attentionHeads / (block.attention.headsPerKVGroup ?? 1)
        }
        if !configuration.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                configuration.hiddenSize,
                configuration.vocabularySize,
                bias: false
            )
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
        let hiddenState = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hiddenState), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        (0 ..< configuration.realAttentionLayerCount).map { _ in KVCacheSimple() }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.contains(".rotary_emb.inv_freq")
                && (!configuration.tieWordEmbeddings || !Self.isOutputHead(key))
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer in
            guard let attention = layer.attentionBlock as? NemotronNASAttention else {
                return nil
            }
            return (attention, ["q_proj", "v_proj"])
        }
    }

    private static func isOutputHead(_ key: String) -> Bool {
        key == "lm_head.weight" || key == "lm_head.scales" || key == "lm_head.biases"
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        if let lmHead {
            return lmHead(hiddenStates)
        }
        return model.embedTokens.asLinear(hiddenStates)
    }
}
