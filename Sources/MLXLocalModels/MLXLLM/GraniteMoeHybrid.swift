import Foundation
import MLX
import MLXNN

// MARK: - Configuration

internal enum GraniteMoeHybridLayerKind: String, Codable, Sendable, Equatable {
    case mamba
    case attention
}

internal struct GraniteMoeHybridConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var maxPositionEmbeddings: Int
    var attentionHeads: Int
    var kvHeads: Int
    var attentionBias: Bool
    var embeddingMultiplier: Float
    var attentionMultiplier: Float
    var logitsScaling: Float
    var residualMultiplier: Float
    var layerTypes: [GraniteMoeHybridLayerKind]
    var rmsNormEps: Float
    var ropeTheta: Float
    var numLocalExperts: Int?
    var numExpertsPerToken: Int?
    var sharedIntermediateSize: Int?
    var mambaHeads: Int?
    var mambaHeadDim: Int?
    var mambaProjBias: Bool?
    var mambaStateDim: Int?
    var mambaConvKernel: Int?
    var mambaGroups: Int?
    var mambaConvBias: Bool?
    var mlpBias: Bool
    var positionEmbeddingType: String
    var tieWordEmbeddings: Bool
    private let timeStepLimitValues: [Float]?

    internal init(
        modelType: String = "granitemoehybrid",
        vocabularySize: Int = 49_152,
        hiddenSize: Int = 2_048,
        intermediateSize: Int = 8_192,
        hiddenLayers: Int = 32,
        maxPositionEmbeddings: Int = 131_072,
        attentionHeads: Int = 32,
        kvHeads: Int = 8,
        attentionBias: Bool = false,
        embeddingMultiplier: Float = 1,
        attentionMultiplier: Float = 1,
        logitsScaling: Float = 1,
        residualMultiplier: Float = 1,
        layerTypes: [GraniteMoeHybridLayerKind]? = nil,
        rmsNormEps: Float = 1e-5,
        ropeTheta: Float = 10_000,
        numLocalExperts: Int? = nil,
        numExpertsPerToken: Int? = nil,
        sharedIntermediateSize: Int? = nil,
        mambaHeads: Int? = nil,
        mambaHeadDim: Int? = nil,
        mambaProjBias: Bool? = nil,
        mambaStateDim: Int? = nil,
        mambaConvKernel: Int? = nil,
        mambaGroups: Int? = nil,
        mambaConvBias: Bool? = nil,
        mlpBias: Bool = false,
        positionEmbeddingType: String = "rope",
        tieWordEmbeddings: Bool = true,
        timeStepLimit: [Float]? = nil
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.attentionBias = attentionBias
        self.embeddingMultiplier = embeddingMultiplier
        self.attentionMultiplier = attentionMultiplier
        self.logitsScaling = logitsScaling
        self.residualMultiplier = residualMultiplier
        self.layerTypes = layerTypes ?? Array(repeating: .attention, count: hiddenLayers)
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.numLocalExperts = numLocalExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.sharedIntermediateSize = sharedIntermediateSize
        self.mambaHeads = mambaHeads
        self.mambaHeadDim = mambaHeadDim
        self.mambaProjBias = mambaProjBias
        self.mambaStateDim = mambaStateDim
        self.mambaConvKernel = mambaConvKernel
        self.mambaGroups = mambaGroups
        self.mambaConvBias = mambaConvBias
        self.mlpBias = mlpBias
        self.positionEmbeddingType = positionEmbeddingType
        self.tieWordEmbeddings = tieWordEmbeddings
        self.timeStepLimitValues = timeStepLimit
    }

    var timeStepMinimum: Float { timeStepLimitValues?.first ?? 0.001 }
    var timeStepMaximum: Float { timeStepLimitValues?.last ?? 100.0 }

    var usesMoE: Bool {
        (numLocalExperts ?? 0) > 0 && (numExpertsPerToken ?? 0) > 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case embeddingMultiplier = "embedding_multiplier"
        case attentionMultiplier = "attention_multiplier"
        case logitsScaling = "logits_scaling"
        case residualMultiplier = "residual_multiplier"
        case layerTypes = "layer_types"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case numLocalExperts = "num_local_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case sharedIntermediateSize = "shared_intermediate_size"
        case mambaHeads = "mamba_n_heads"
        case mambaHeadDim = "mamba_d_head"
        case mambaProjBias = "mamba_proj_bias"
        case mambaStateDim = "mamba_d_state"
        case mambaConvKernel = "mamba_d_conv"
        case mambaGroups = "mamba_n_groups"
        case mambaConvBias = "mamba_conv_bias"
        case mlpBias = "mlp_bias"
        case positionEmbeddingType = "position_embedding_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case timeStepLimitValues = "time_step_limit"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        let layerTypes = try Self.decodeLayerTypes(from: container, hiddenLayers: hiddenLayers)

        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "granitemoehybrid",
            vocabularySize: try container.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 49_152,
            hiddenSize: try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2_048,
            intermediateSize: try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? 8_192,
            hiddenLayers: hiddenLayers,
            maxPositionEmbeddings: try container.decodeIfPresent(
                Int.self,
                forKey: .maxPositionEmbeddings
            ) ?? 131_072,
            attentionHeads: try container.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? 32,
            kvHeads: try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8,
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? false,
            embeddingMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .embeddingMultiplier
            ) ?? 1,
            attentionMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .attentionMultiplier
            ) ?? 1,
            logitsScaling: try container.decodeIfPresent(Float.self, forKey: .logitsScaling)
                ?? 1,
            residualMultiplier: try container.decodeIfPresent(
                Float.self,
                forKey: .residualMultiplier
            ) ?? 1,
            layerTypes: layerTypes,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000,
            numLocalExperts: try container.decodeIfPresent(Int.self, forKey: .numLocalExperts),
            numExpertsPerToken: try container.decodeIfPresent(
                Int.self,
                forKey: .numExpertsPerToken
            ),
            sharedIntermediateSize: try container.decodeIfPresent(
                Int.self,
                forKey: .sharedIntermediateSize
            ),
            mambaHeads: try container.decodeIfPresent(Int.self, forKey: .mambaHeads),
            mambaHeadDim: try container.decodeIfPresent(Int.self, forKey: .mambaHeadDim),
            mambaProjBias: try container.decodeIfPresent(Bool.self, forKey: .mambaProjBias),
            mambaStateDim: try container.decodeIfPresent(Int.self, forKey: .mambaStateDim),
            mambaConvKernel: try container.decodeIfPresent(Int.self, forKey: .mambaConvKernel),
            mambaGroups: try container.decodeIfPresent(Int.self, forKey: .mambaGroups),
            mambaConvBias: try container.decodeIfPresent(Bool.self, forKey: .mambaConvBias),
            mlpBias: try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false,
            positionEmbeddingType: try container.decodeIfPresent(
                String.self,
                forKey: .positionEmbeddingType
            ) ?? "rope",
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true,
            timeStepLimit: try container.decodeIfPresent(
                [Float].self,
                forKey: .timeStepLimitValues
            )
        )
    }

    private static func decodeLayerTypes(
        from container: KeyedDecodingContainer<CodingKeys>,
        hiddenLayers: Int
    ) throws -> [GraniteMoeHybridLayerKind] {
        guard let names = try container.decodeIfPresent([String].self, forKey: .layerTypes) else {
            return Array(repeating: .attention, count: hiddenLayers)
        }
        guard names.count == hiddenLayers else {
            throw DecodingError.dataCorruptedError(
                forKey: .layerTypes,
                in: container,
                debugDescription: "layer_types count must match num_hidden_layers"
            )
        }
        return try names.map { name in
            guard let kind = GraniteMoeHybridLayerKind(rawValue: name) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .layerTypes,
                    in: container,
                    debugDescription: "Unsupported Granite MoE Hybrid layer type: \(name)"
                )
            }
            return kind
        }
    }
}

// MARK: - Plans

internal struct GraniteMoeHybridLayerPlan: Sendable, Equatable {
    internal let kinds: [GraniteMoeHybridLayerKind]
    internal let firstAttentionIndex: Int?
    internal let firstMambaIndex: Int?

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        precondition(configuration.hiddenLayers > 0, "Granite MoE Hybrid must have layers")
        precondition(
            configuration.layerTypes.count == configuration.hiddenLayers,
            "Granite MoE Hybrid layer_types count must match hidden layer count"
        )
        self.kinds = configuration.layerTypes
        self.firstAttentionIndex = kinds.firstIndex(of: .attention)
        self.firstMambaIndex = kinds.firstIndex(of: .mamba)
    }

    internal var count: Int {
        kinds.count
    }

    internal func kind(at index: Int) -> GraniteMoeHybridLayerKind {
        kinds[index]
    }
}

internal struct GraniteMoeHybridAttentionLayout: Sendable, Equatable {
    internal let hiddenSize: Int
    internal let attentionHeads: Int
    internal let keyValueHeads: Int
    internal let headDimensions: Int
    internal let queryDimensions: Int
    internal let keyValueDimensions: Int
    internal let scale: Float
    internal let usesRotaryPosition: Bool

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        precondition(configuration.hiddenSize > 0, "Granite MoE Hybrid hidden size is invalid")
        precondition(
            configuration.attentionHeads > 0,
            "Granite MoE Hybrid attention head count is invalid"
        )
        precondition(configuration.kvHeads > 0, "Granite MoE Hybrid KV head count is invalid")
        precondition(
            configuration.hiddenSize.isMultiple(of: configuration.attentionHeads),
            "Granite MoE Hybrid hidden size must divide by attention heads"
        )
        precondition(
            configuration.attentionHeads.isMultiple(of: configuration.kvHeads),
            "Granite MoE Hybrid attention heads must group KV heads"
        )

        self.hiddenSize = configuration.hiddenSize
        self.attentionHeads = configuration.attentionHeads
        self.keyValueHeads = configuration.kvHeads
        self.headDimensions = configuration.hiddenSize / configuration.attentionHeads
        self.queryDimensions = attentionHeads * headDimensions
        self.keyValueDimensions = keyValueHeads * headDimensions
        self.scale = configuration.attentionMultiplier
        self.usesRotaryPosition = configuration.positionEmbeddingType != "nope"
    }
}

internal struct GraniteMoeHybridMambaLayout: Sendable {
    internal let hiddenSize: Int
    internal let heads: Int
    internal let headDimensions: Int
    internal let stateDimensions: Int
    internal let groups: Int
    internal let convolutionKernelSize: Int
    internal let intermediateSize: Int
    internal let convolutionDimensions: Int
    internal let projectionDimensions: Int
    internal let timeStepLimit: (Float, Float)

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        guard let heads = configuration.mambaHeads,
            let headDimensions = configuration.mambaHeadDim,
            let stateDimensions = configuration.mambaStateDim,
            let convolutionKernelSize = configuration.mambaConvKernel,
            let groups = configuration.mambaGroups
        else {
            preconditionFailure("Granite MoE Hybrid Mamba layer requires complete Mamba config")
        }

        precondition(heads > 0, "Granite MoE Hybrid Mamba heads must be positive")
        precondition(headDimensions > 0, "Granite MoE Hybrid Mamba head dim must be positive")
        precondition(stateDimensions > 0, "Granite MoE Hybrid Mamba state dim must be positive")
        precondition(groups > 0, "Granite MoE Hybrid Mamba groups must be positive")
        precondition(
            convolutionKernelSize > 0,
            "Granite MoE Hybrid Mamba convolution kernel must be positive"
        )

        self.hiddenSize = configuration.hiddenSize
        self.heads = heads
        self.headDimensions = headDimensions
        self.stateDimensions = stateDimensions
        self.groups = groups
        self.convolutionKernelSize = convolutionKernelSize
        self.intermediateSize = heads * headDimensions
        self.convolutionDimensions = intermediateSize + 2 * groups * stateDimensions
        self.projectionDimensions = intermediateSize + convolutionDimensions + heads
        self.timeStepLimit = (configuration.timeStepMinimum, configuration.timeStepMaximum)
    }
}

internal struct GraniteMoeHybridMoEPlan: Sendable, Equatable {
    internal let expertCount: Int
    internal let selectedExpertCount: Int
    internal let sharedIntermediateSize: Int

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        guard let expertCount = configuration.numLocalExperts,
            let selectedExpertCount = configuration.numExpertsPerToken,
            let sharedIntermediateSize = configuration.sharedIntermediateSize
        else {
            preconditionFailure("Granite MoE Hybrid MoE layer requires complete MoE config")
        }

        precondition(expertCount > 0, "Granite MoE Hybrid expert count must be positive")
        precondition(selectedExpertCount > 0, "Granite MoE Hybrid top-k must be positive")
        precondition(
            selectedExpertCount <= expertCount,
            "Granite MoE Hybrid top-k cannot exceed expert count"
        )
        precondition(
            sharedIntermediateSize > 0,
            "Granite MoE Hybrid shared intermediate size must be positive"
        )

        self.expertCount = expertCount
        self.selectedExpertCount = selectedExpertCount
        self.sharedIntermediateSize = sharedIntermediateSize
    }

    internal func route(_ logits: MLXArray) -> (indices: MLXArray, gates: MLXArray) {
        let indices = argPartition(-logits, kth: selectedExpertCount - 1, axis: -1)[
            .ellipsis,
            ..<selectedExpertCount
        ]
        let selected = takeAlong(logits, indices, axis: -1)
        return (indices, softmax(selected, axis: -1, precise: true))
    }
}

internal struct GraniteMoeHybridSanitizerPlan: Sendable {
    internal let configuration: GraniteMoeHybridConfiguration

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        self.configuration = configuration
    }

    internal func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = MLXQuantizedWeightSanitizer.sanitize(
            weights,
            strategy: .automatic(),
            sidecarPolicy: .dropActivationScale
        ).weights

        sanitized = sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for (key, value) in sanitized where key.contains("conv1d.weight") && value.dim(-1) != 1 {
            sanitized[key] = value.swappedAxes(1, 2)
        }

        if configuration.usesMoE {
            sanitized = remapBlockSparseMoEWeights(sanitized)
        } else {
            sanitized = remapDenseSharedMLPWeights(sanitized)
        }

        return sanitized
    }

    private func remapBlockSparseMoEWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var remapped = weights
        for layerIndex in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layerIndex).block_sparse_moe"
            remapSplitInput(
                in: &remapped,
                sourcePrefix: prefix,
                destinationPrefix: "\(prefix).switch_mlp",
                splitAxis: 1
            )
        }
        return remapped
    }

    private func remapDenseSharedMLPWeights(
        _ weights: [String: MLXArray]
    ) -> [String: MLXArray] {
        var remapped = weights
        for layerIndex in 0 ..< configuration.hiddenLayers {
            remapSplitInput(
                in: &remapped,
                sourcePrefix: "model.layers.\(layerIndex).shared_mlp",
                destinationPrefix: "model.layers.\(layerIndex).mlp",
                splitAxis: 0
            )
        }
        return remapped
    }

    private func remapSplitInput(
        in weights: inout [String: MLXArray],
        sourcePrefix: String,
        destinationPrefix: String,
        splitAxis: Int
    ) {
        guard let inputWeight = weights.removeValue(forKey: "\(sourcePrefix).input_linear.weight")
        else {
            return
        }

        let inputParts = inputWeight.split(parts: 2, axis: splitAxis)
        weights["\(destinationPrefix).gate_proj.weight"] = inputParts[0]
        weights["\(destinationPrefix).up_proj.weight"] = inputParts[1]

        if let downWeight = weights.removeValue(forKey: "\(sourcePrefix).output_linear.weight") {
            weights["\(destinationPrefix).down_proj.weight"] = downWeight
        }
    }
}

// MARK: - Layers

internal final class GraniteMoeHybridRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray

    private let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self._weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray?) -> MLXArray {
        let gated = gate.map { hiddenStates * silu($0) } ?? hiddenStates
        return MLXFast.rmsNorm(gated, weight: weight, eps: eps)
    }
}

internal final class GraniteMoeHybridMamba2Mixer: Module {
    let layout: GraniteMoeHybridMambaLayout

    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "D") var residualScale: MLXArray
    @ModuleInfo(key: "norm") var norm: GraniteMoeHybridRMSNormGated

    init(_ configuration: GraniteMoeHybridConfiguration) {
        self.layout = GraniteMoeHybridMambaLayout(configuration)

        _conv1d.wrappedValue = Conv1d(
            inputChannels: layout.convolutionDimensions,
            outputChannels: layout.convolutionDimensions,
            kernelSize: layout.convolutionKernelSize,
            groups: layout.convolutionDimensions,
            bias: configuration.mambaConvBias ?? false
        )
        _inProj.wrappedValue = Linear(
            layout.hiddenSize,
            layout.projectionDimensions,
            bias: configuration.mambaProjBias ?? false
        )
        _outProj.wrappedValue = Linear(
            layout.intermediateSize,
            layout.hiddenSize,
            bias: configuration.mambaProjBias ?? false
        )
        _dtBias.wrappedValue = MLXArray.ones([layout.heads])
        _aLog.wrappedValue = log((MLXArray(0 ..< layout.heads).asType(.float32) + 1))
        _residualScale.wrappedValue = MLXArray.ones([layout.heads])
        _norm.wrappedValue = GraniteMoeHybridRMSNormGated(
            dimensions: layout.intermediateSize,
            eps: configuration.rmsNormEps
        )
        super.init()
    }

    private func applyConvolution(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let batchSize = input.dim(0)
        let stateLength = max(0, layout.convolutionKernelSize - 1)
        let state = cache?[0] ?? MLXArray.zeros(
            [batchSize, stateLength, layout.convolutionDimensions],
            dtype: input.dtype
        )
        let padded = concatenated([state, input], axis: 1)

        if let cache {
            let end = padded.dim(1)
            let start = max(0, end - stateLength)
            cache[0] = padded[0..., start ..< end, 0...]
        }

        return silu(conv1d(padded))
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXArray?,
        cache: MambaCache?
    ) -> MLXArray {
        let projected = inProj(hiddenStates)
        let splits = split(
            projected,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.convolutionDimensions,
            ],
            axis: -1
        )
        let gate = splits[0]
        var convInput = splits[1]
        let timeSteps = splits[2]

        if let mask {
            convInput = MLX.where(
                expandedDimensions(mask, axis: -1),
                convInput,
                MLXArray.zeros(like: convInput)
            )
        }

        let convOutput = applyConvolution(convInput, cache: cache)
        let convSplits = split(
            convOutput,
            indices: [
                layout.intermediateSize,
                layout.intermediateSize + layout.groups * layout.stateDimensions,
            ],
            axis: -1
        )

        let hidden = convSplits[0].reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            layout.heads,
            layout.headDimensions
        )
        let inputState = convSplits[1].reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            layout.groups,
            layout.stateDimensions
        )
        let outputState = convSplits[2].reshaped(
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            layout.groups,
            layout.stateDimensions
        )
        let dt = timeSteps.reshaped(hiddenStates.dim(0), hiddenStates.dim(1), layout.heads)

        let (output, nextState) = ssmUpdate(
            hiddenStates: hidden,
            ALog: aLog,
            B: inputState,
            C: outputState,
            D: residualScale,
            dt: dt,
            dtBias: dtBias,
            state: cache?[1],
            timeStepLimit: layout.timeStepLimit,
            mask: mask
        )

        if let cache {
            cache[1] = nextState
        }

        let normalized = norm(output.flattened(start: 2), gate: gate)
        return outProj(normalized)
    }
}

internal final class GraniteMoeHybridAttention: Module {
    let layout: GraniteMoeHybridAttentionLayout

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer?

    init(_ configuration: GraniteMoeHybridConfiguration) {
        self.layout = GraniteMoeHybridAttentionLayout(configuration)

        _wq.wrappedValue = Linear(
            layout.hiddenSize,
            layout.queryDimensions,
            bias: configuration.attentionBias
        )
        _wk.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueDimensions,
            bias: configuration.attentionBias
        )
        _wv.wrappedValue = Linear(
            layout.hiddenSize,
            layout.keyValueDimensions,
            bias: configuration.attentionBias
        )
        _wo.wrappedValue = Linear(
            layout.queryDimensions,
            layout.hiddenSize,
            bias: configuration.attentionBias
        )

        if layout.usesRotaryPosition {
            self.rope = initializeRope(
                dims: layout.headDimensions,
                base: configuration.ropeTheta,
                traditional: false,
                scalingConfig: nil,
                maxPositionEmbeddings: configuration.maxPositionEmbeddings
            )
        } else {
            self.rope = nil
        }
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = x.dim(0)
        let tokenCount = x.dim(1)

        var queries = wq(x)
            .reshaped(batchSize, tokenCount, layout.attentionHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        var keys = wk(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)
        let values = wv(x)
            .reshaped(batchSize, tokenCount, layout.keyValueHeads, layout.headDimensions)
            .transposed(0, 2, 1, 3)

        if let rope {
            queries = applyRotaryPosition(rope, to: queries, cache: cache)
            keys = applyRotaryPosition(rope, to: keys, cache: cache)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return wo(output)
    }
}

internal final class GraniteMoeHybridTopKGating: Module {
    let plan: GraniteMoeHybridMoEPlan

    @ModuleInfo(key: "layer") var layer: Linear

    init(inputSize: Int, plan: GraniteMoeHybridMoEPlan) {
        self.plan = plan
        _layer.wrappedValue = Linear(inputSize, plan.expertCount, bias: false)
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let route = plan.route(layer(hiddenStates))
        return (route.indices, route.gates)
    }
}

internal final class GraniteMoeHybridMoE: Module, UnaryLayer {
    let plan: GraniteMoeHybridMoEPlan

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "router") var router: GraniteMoeHybridTopKGating

    init(_ configuration: GraniteMoeHybridConfiguration) {
        self.plan = GraniteMoeHybridMoEPlan(configuration)

        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: configuration.hiddenSize,
            hiddenDims: configuration.intermediateSize,
            numExperts: plan.expertCount,
            bias: false
        )
        _router.wrappedValue = GraniteMoeHybridTopKGating(
            inputSize: configuration.hiddenSize,
            plan: plan
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, gates) = router(x)
        return (switchMLP(x, indices) * gates[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

internal final class GraniteMoeHybridSharedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "input_linear") var inputLinear: Linear
    @ModuleInfo(key: "output_linear") var outputLinear: Linear

    init(_ configuration: GraniteMoeHybridConfiguration) {
        let plan = GraniteMoeHybridMoEPlan(configuration)
        _inputLinear.wrappedValue = Linear(
            configuration.hiddenSize,
            plan.sharedIntermediateSize * 2,
            bias: false
        )
        _outputLinear.wrappedValue = Linear(
            plan.sharedIntermediateSize,
            configuration.hiddenSize,
            bias: false
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let splits = inputLinear(x).split(parts: 2, axis: -1)
        return outputLinear(silu(splits[0]) * splits[1])
    }
}

internal final class GraniteMoeHybridMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(_ configuration: GraniteMoeHybridConfiguration) {
        _gate.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: configuration.mlpBias
        )
        _down.wrappedValue = Linear(
            configuration.intermediateSize,
            configuration.hiddenSize,
            bias: configuration.mlpBias
        )
        _up.wrappedValue = Linear(
            configuration.hiddenSize,
            configuration.intermediateSize,
            bias: configuration.mlpBias
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

internal final class GraniteMoeHybridLayer: Module {
    let layerKind: GraniteMoeHybridLayerKind
    let residualMultiplier: Float
    let usesMoE: Bool

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: GraniteMoeHybridAttention?
    @ModuleInfo(key: "mamba") var mamba: GraniteMoeHybridMamba2Mixer?
    @ModuleInfo(key: "block_sparse_moe") var blockSparseMoE: GraniteMoeHybridMoE?
    @ModuleInfo(key: "shared_mlp") var sharedMLP: GraniteMoeHybridSharedMLP?
    @ModuleInfo(key: "mlp") var mlp: GraniteMoeHybridMLP?

    init(
        _ configuration: GraniteMoeHybridConfiguration,
        layerIndex: Int,
        layerPlan: GraniteMoeHybridLayerPlan
    ) {
        self.layerKind = layerPlan.kind(at: layerIndex)
        self.residualMultiplier = configuration.residualMultiplier
        self.usesMoE = configuration.usesMoE

        switch layerKind {
        case .mamba:
            _mamba.wrappedValue = GraniteMoeHybridMamba2Mixer(configuration)
        case .attention:
            _selfAttention.wrappedValue = GraniteMoeHybridAttention(configuration)
        }

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )

        if usesMoE {
            _blockSparseMoE.wrappedValue = GraniteMoeHybridMoE(configuration)
            _sharedMLP.wrappedValue = GraniteMoeHybridSharedMLP(configuration)
        } else {
            _mlp.wrappedValue = GraniteMoeHybridMLP(configuration)
        }
        super.init()
    }

    var isMamba: Bool {
        layerKind == .mamba
    }

    func callAsFunction(
        _ x: MLXArray,
        attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        ssmMask: MLXArray?,
        cache: KVCache?
    ) -> MLXArray {
        let hidden: MLXArray
        switch layerKind {
        case .mamba:
            guard let mamba else {
                preconditionFailure("Granite MoE Hybrid Mamba layer is missing its mixer")
            }
            hidden = mamba(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache)
        case .attention:
            guard let selfAttention else {
                preconditionFailure("Granite MoE Hybrid attention layer is missing attention")
            }
            hidden = selfAttention(inputLayerNorm(x), mask: attentionMask, cache: cache)
        }

        let residual = x + hidden * residualMultiplier
        let feedForwardInput = postAttentionLayerNorm(residual)

        let feedForwardOutput: MLXArray
        if usesMoE {
            guard let blockSparseMoE, let sharedMLP else {
                preconditionFailure("Granite MoE Hybrid MoE layer is missing feed-forward modules")
            }
            feedForwardOutput = blockSparseMoE(feedForwardInput) + sharedMLP(feedForwardInput)
        } else {
            guard let mlp else {
                preconditionFailure("Granite MoE Hybrid dense layer is missing MLP")
            }
            feedForwardOutput = mlp(feedForwardInput)
        }

        return residual + feedForwardOutput * residualMultiplier
    }
}

// MARK: - Model

internal final class GraniteMoeHybridModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [GraniteMoeHybridLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    let layerPlan: GraniteMoeHybridLayerPlan
    let embeddingMultiplier: Float

    init(_ configuration: GraniteMoeHybridConfiguration) {
        precondition(
            configuration.vocabularySize > 0,
            "Granite MoE Hybrid vocabulary size must be positive"
        )

        let plan = GraniteMoeHybridLayerPlan(configuration)
        self.layerPlan = plan
        self.embeddingMultiplier = configuration.embeddingMultiplier

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: configuration.vocabularySize,
            dimensions: configuration.hiddenSize
        )
        _layers.wrappedValue = (0 ..< plan.count).map { index in
            GraniteMoeHybridLayer(configuration, layerIndex: index, layerPlan: plan)
        }
        _norm.wrappedValue = RMSNorm(
            dimensions: configuration.hiddenSize,
            eps: configuration.rmsNormEps
        )
        super.init()
    }

    func hiddenStates(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        var hidden = embedTokens(inputs) * embeddingMultiplier
        let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
        if let index = layerPlan.firstAttentionIndex {
            attentionMask = createAttentionMask(
                h: hidden,
                cache: cacheValue(in: cacheArray, at: index)
            )
        } else {
            attentionMask = .none
        }

        let ssmMask: MLXArray?
        if let index = layerPlan.firstMambaIndex {
            ssmMask = createSSMMask(
                h: hidden,
                cache: cacheValue(in: cacheArray, at: index) as? MambaCache
            )
        } else {
            ssmMask = nil
        }

        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                attentionMask: layer.isMamba ? .none : attentionMask,
                ssmMask: layer.isMamba ? ssmMask : nil,
                cache: cacheValue(in: cacheArray, at: index)
            )
        }

        return hidden
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache?]? = nil) -> MLXArray {
        norm(hiddenStates(inputs, cache: cache))
    }

    private func cacheValue(in cache: [KVCache?], at index: Int) -> KVCache? {
        cache.indices.contains(index) ? cache[index] : nil
    }
}

internal final class GraniteMoeHybridModel: Module, LLMModel, KVCacheDimensionProvider,
    GreedyTokenModel {
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    @ModuleInfo(key: "model") var model: GraniteMoeHybridModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: GraniteMoeHybridConfiguration
    let logitsScaling: Float

    internal init(_ configuration: GraniteMoeHybridConfiguration) {
        self.configuration = configuration
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = configuration.layerTypes.map { kind in
            kind == .attention ? configuration.kvHeads : 0
        }
        self.logitsScaling = configuration.logitsScaling
        _model.wrappedValue = GraniteMoeHybridModelInner(configuration)

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

    internal func hiddenStates(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.hiddenStates(inputs, cache: cache)
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hidden = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: logits(from: hidden), state: state)
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        configuration.layerTypes.map { kind in
            switch kind {
            case .mamba:
                MambaCache()
            case .attention:
                KVCacheSimple()
            }
        }
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        GraniteMoeHybridSanitizerPlan(configuration).sanitize(weights)
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        let logits: MLXArray
        if let lmHead {
            logits = lmHead(hiddenStates)
        } else {
            logits = model.embedTokens.asLinear(hiddenStates)
        }
        return logits / logitsScaling
    }
}

// MARK: - LoRA

extension GraniteMoeHybridModel: LoRAModel {
    internal var loraLayers: [Module] {
        model.layers
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.compactMap { layer in
            guard let attention = layer.selfAttention else {
                return nil
            }
            return (attention, ["q_proj", "v_proj"])
        }
    }
}
