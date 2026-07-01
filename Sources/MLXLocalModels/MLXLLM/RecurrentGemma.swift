import Foundation
import MLX
import MLXFast
import MLXNN

internal struct RecurrentGemmaConfiguration: Codable, Equatable, Sendable {
    var modelType: String
    var attentionBias: Bool
    var convolutionWidth: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var logitsSoftCap: Float
    var attentionHeads: Int
    var hiddenLayers: Int
    var keyValueHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var attentionWindowSize: Int
    var vocabularySize: Int
    var scaleEmbeddingsBySqrtDim: Bool
    var blockTypes: [String]
    var tieWordEmbeddings: Bool

    internal init(
        modelType: String = "recurrent_gemma",
        attentionBias: Bool = true,
        convolutionWidth: Int = 4,
        hiddenSize: Int,
        intermediateSize: Int,
        logitsSoftCap: Float = 30,
        attentionHeads: Int,
        hiddenLayers: Int,
        keyValueHeads: Int = 1,
        rmsNormEps: Float = 1e-6,
        ropeTheta: Float = 10_000,
        attentionWindowSize: Int = 2_048,
        vocabularySize: Int,
        scaleEmbeddingsBySqrtDim: Bool = true,
        blockTypes: [String] = ["recurrent", "attention"],
        tieWordEmbeddings: Bool = true
    ) {
        precondition(hiddenSize > 0, "RecurrentGemma hidden size must be positive")
        precondition(hiddenLayers > 0, "RecurrentGemma layer count must be positive")
        precondition(attentionHeads > 0, "RecurrentGemma attention head count must be positive")
        precondition(
            hiddenSize.isMultiple(of: attentionHeads),
            "RecurrentGemma hidden size must divide evenly across attention heads"
        )
        precondition(!blockTypes.isEmpty, "RecurrentGemma block types cannot be empty")

        self.modelType = modelType
        self.attentionBias = attentionBias
        self.convolutionWidth = convolutionWidth
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.logitsSoftCap = logitsSoftCap
        self.attentionHeads = attentionHeads
        self.hiddenLayers = hiddenLayers
        self.keyValueHeads = keyValueHeads
        self.rmsNormEps = rmsNormEps
        self.ropeTheta = ropeTheta
        self.attentionWindowSize = attentionWindowSize
        self.vocabularySize = vocabularySize
        self.scaleEmbeddingsBySqrtDim = scaleEmbeddingsBySqrtDim
        self.blockTypes = blockTypes
        self.tieWordEmbeddings = tieWordEmbeddings
    }

    var headDim: Int {
        hiddenSize / attentionHeads
    }

    func blockType(for layerIndex: Int) -> String {
        blockTypes[layerIndex % blockTypes.count]
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case attentionBias = "attention_bias"
        case convolutionWidth = "conv1d_width"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case logitsSoftCap = "logits_soft_cap"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case keyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionWindowSize = "attention_window_size"
        case vocabularySize = "vocab_size"
        case scaleEmbeddingsBySqrtDim = "embeddings_scale_by_sqrt_dim"
        case blockTypes = "block_types"
        case underscoredBlockTypes = "_block_types"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let blockTypes = try container.decodeIfPresent([String].self, forKey: .blockTypes)
            ?? container.decodeIfPresent([String].self, forKey: .underscoredBlockTypes)
            ?? ["recurrent", "attention"]
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "recurrent_gemma",
            attentionBias: try container.decodeIfPresent(Bool.self, forKey: .attentionBias)
                ?? true,
            convolutionWidth: try container.decodeIfPresent(Int.self, forKey: .convolutionWidth)
                ?? 4,
            hiddenSize: try container.decode(Int.self, forKey: .hiddenSize),
            intermediateSize: try container.decode(Int.self, forKey: .intermediateSize),
            logitsSoftCap: try container.decodeIfPresent(Float.self, forKey: .logitsSoftCap)
                ?? 30,
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            hiddenLayers: try container.decode(Int.self, forKey: .hiddenLayers),
            keyValueHeads: try container.decodeIfPresent(Int.self, forKey: .keyValueHeads) ?? 1,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-6,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 10_000,
            attentionWindowSize: try container.decodeIfPresent(
                Int.self,
                forKey: .attentionWindowSize
            ) ?? 2_048,
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            scaleEmbeddingsBySqrtDim: try container.decodeIfPresent(
                Bool.self,
                forKey: .scaleEmbeddingsBySqrtDim
            ) ?? true,
            blockTypes: blockTypes,
            tieWordEmbeddings: try container.decodeIfPresent(
                Bool.self,
                forKey: .tieWordEmbeddings
            ) ?? true
        )
    }

    internal func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(modelType, forKey: .modelType)
        try container.encode(attentionBias, forKey: .attentionBias)
        try container.encode(convolutionWidth, forKey: .convolutionWidth)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(logitsSoftCap, forKey: .logitsSoftCap)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(keyValueHeads, forKey: .keyValueHeads)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(ropeTheta, forKey: .ropeTheta)
        try container.encode(attentionWindowSize, forKey: .attentionWindowSize)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(scaleEmbeddingsBySqrtDim, forKey: .scaleEmbeddingsBySqrtDim)
        try container.encode(blockTypes, forKey: .blockTypes)
        try container.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
    }
}

internal struct RecurrentGemmaLayerPlan: Equatable, Sendable {
    let blockTypes: [String]

    init(_ config: RecurrentGemmaConfiguration) {
        self.blockTypes = (0 ..< config.hiddenLayers).map { config.blockType(for: $0) }
    }

    var firstAttentionIndex: Int? {
        blockTypes.firstIndex(of: "attention")
    }

    var cacheKinds: [String] {
        blockTypes.map { $0 == "attention" ? "attention" : "recurrent" }
    }
}

private final class RecurrentGemmaRMSNorm: Module, UnaryLayer {
    private let eps: Float
    @ParameterInfo(key: "weight") private var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(
            input,
            weight: weight + MLXArray(1, dtype: weight.dtype),
            eps: eps
        )
    }
}

private func recurrentGemmaScan(
    input: MLXArray,
    decay: MLXArray,
    state: MLXArray?
) -> (MLXArray, MLXArray) {
    let batchSize = input.dim(0)
    let tokenCount = input.dim(1)
    let width = input.dim(2)

    var runningState = state ?? MLXArray.zeros([batchSize, width], dtype: input.dtype)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(tokenCount)

    for index in 0 ..< tokenCount {
        runningState = decay[0..., index, 0...] * runningState + input[0..., index, 0...]
        outputs.append(runningState[0..., .newAxis, 0...])
    }

    return (concatenated(outputs, axis: 1), runningState)
}

private final class RecurrentGemmaRGLRU: Module {
    private let width: Int
    private let headCount: Int
    private let headDim: Int

    @ParameterInfo(key: "recurrent_param") private var recurrentParameter: MLXArray
    @ParameterInfo(key: "input_gate_weight") private var inputGateWeight: MLXArray
    @ParameterInfo(key: "input_gate_bias") private var inputGateBias: MLXArray
    @ParameterInfo(key: "recurrent_gate_weight") private var recurrentGateWeight: MLXArray
    @ParameterInfo(key: "recurrent_gate_bias") private var recurrentGateBias: MLXArray

    init(width: Int, headCount: Int) {
        precondition(width.isMultiple(of: headCount), "RGLRU width must divide across heads")
        self.width = width
        self.headCount = headCount
        self.headDim = width / headCount
        _recurrentParameter.wrappedValue = MLXArray.zeros([width])
        _inputGateWeight.wrappedValue = MLXArray.zeros([headCount, headDim, headDim])
        _inputGateBias.wrappedValue = MLXArray.zeros([headCount, headDim])
        _recurrentGateWeight.wrappedValue = MLXArray.zeros([headCount, headDim, headDim])
        _recurrentGateBias.wrappedValue = MLXArray.zeros([headCount, headDim])
    }

    func callAsFunction(_ input: MLXArray, cache: MLXArray?) -> (MLXArray, MLXArray) {
        let inputGate = applyBlockLinear(
            input,
            weight: inputGateWeight,
            bias: inputGateBias
        )
        let recurrentGate = applyBlockLinear(
            input,
            weight: recurrentGateWeight,
            bias: recurrentGateBias
        )

        let logDecay = MLXArray(-8, dtype: input.dtype)
            * recurrentGate
            * softplus(recurrentParameter).asType(input.dtype)
        let decay = exp(logDecay)
        let decaySquared = exp(MLXArray(2, dtype: input.dtype) * logDecay)
        let gatedInput = input * inputGate
        let multiplier = sqrt(MLXArray(1, dtype: input.dtype) - decaySquared)
        if cache == nil {
            multiplier[0..., 0, 0...] = MLXArray(1, dtype: multiplier.dtype)
        }
        return recurrentGemmaScan(input: gatedInput * multiplier, decay: decay, state: cache)
    }

    private func applyBlockLinear(
        _ input: MLXArray,
        weight: MLXArray,
        bias: MLXArray
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)
        let projected = input
            .reshaped(batchSize, tokenCount, headCount, headDim)
            .swappedAxes(1, 2)
            .matmul(weight)
            .swappedAxes(1, 2) + bias
        return sigmoid(projected.reshaped(batchSize, tokenCount, width))
    }
}

private final class RecurrentGemmaRecurrentBlock: Module {
    private let convolutionWidth: Int
    @ModuleInfo(key: "linear_y") private var yProjection: Linear
    @ModuleInfo(key: "linear_x") private var xProjection: Linear
    @ModuleInfo(key: "linear_out") private var outputProjection: Linear
    @ModuleInfo(key: "conv_1d") private var convolution: Conv1d
    @ModuleInfo(key: "rg_lru") private var rgLRU: RecurrentGemmaRGLRU

    init(_ config: RecurrentGemmaConfiguration) {
        self.convolutionWidth = config.convolutionWidth
        _yProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _xProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _outputProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        _convolution.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convolutionWidth,
            padding: 0,
            groups: config.hiddenSize,
            bias: true
        )
        _rgLRU.wrappedValue = RecurrentGemmaRGLRU(
            width: config.hiddenSize,
            headCount: config.attentionHeads
        )
    }

    func callAsFunction(_ input: MLXArray, cache: MambaCache?) -> MLXArray {
        let gate = geluApproximate(yProjection(input))
        let projected = xProjection(input)
        let convolutionInput = paddedConvolutionInput(projected, state: cache?[0])
        let convolved = geluApproximate(convolution(convolutionInput))
        let newConvolutionState = convolutionInput[0..., (1 - convolutionWidth)..., 0...]
        let (recurrentOutput, newRecurrentState) = rgLRU(convolved, cache: cache?[1])

        if let cache {
            cache[0] = newConvolutionState
            cache[1] = newRecurrentState
            cache.offset += input.dim(1)
        }

        return outputProjection(recurrentOutput * gate)
    }

    private func paddedConvolutionInput(_ input: MLXArray, state: MLXArray?) -> MLXArray {
        if let state {
            return concatenated([state, input], axis: 1)
        }
        return padded(
            input,
            widths: [
                IntOrPair((0, 0)),
                IntOrPair((convolutionWidth - 1, 0)),
                IntOrPair((0, 0))
            ]
        )
    }
}

private final class RecurrentGemmaLocalAttentionBlock: Module {
    private let headCount: Int
    private let headDim: Int
    private let scale: Float
    private let rope: RoPE

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "k_proj") private var keyProjection: Linear
    @ModuleInfo(key: "v_proj") private var valueProjection: Linear
    @ModuleInfo(key: "o_proj") private var outputProjection: Linear

    init(_ config: RecurrentGemmaConfiguration) {
        self.headCount = config.attentionHeads
        self.headDim = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.rope = RoPE(dimensions: config.headDim / 2, traditional: false)
        _queryProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: false
        )
        _keyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.headDim,
            bias: false
        )
        _valueProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.headDim,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize,
            bias: true
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let tokenCount = input.dim(1)
        let offset = cache?.offset ?? 0

        let queries = rope(
            queryProjection(input)
                .reshaped(batchSize, tokenCount, headCount, headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let keys = rope(
            keyProjection(input)
                .reshaped(batchSize, tokenCount, 1, headDim)
                .transposed(0, 2, 1, 3),
            offset: offset
        )
        let values = valueProjection(input)
            .reshaped(batchSize, tokenCount, 1, headDim)
            .transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, tokenCount, -1)

        return outputProjection(output)
    }
}

private final class RecurrentGemmaMLPBlock: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") private var upProjection: Linear
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear

    init(_ config: RecurrentGemmaConfiguration) {
        let expandedHalf = config.intermediateSize / 2
        _upProjection.wrappedValue = Linear(config.hiddenSize, expandedHalf, bias: true)
        _gateProjection.wrappedValue = Linear(config.hiddenSize, expandedHalf, bias: true)
        _downProjection.wrappedValue = Linear(expandedHalf, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(geluApproximate(gateProjection(input)) * upProjection(input))
    }
}

private final class RecurrentGemmaResidualBlock: Module {
    let temporalBlockType: String

    @ModuleInfo(key: "temporal_pre_norm") private var temporalPreNorm: RecurrentGemmaRMSNorm
    @ModuleInfo(key: "temporal_block") fileprivate var temporalBlock: Module
    @ModuleInfo(key: "channel_pre_norm") private var channelPreNorm: RecurrentGemmaRMSNorm
    @ModuleInfo(key: "mlp_block") fileprivate var mlpBlock: RecurrentGemmaMLPBlock

    init(_ config: RecurrentGemmaConfiguration, layerIndex: Int) {
        self.temporalBlockType = config.blockType(for: layerIndex)
        _temporalPreNorm.wrappedValue = RecurrentGemmaRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        if temporalBlockType == "recurrent" {
            _temporalBlock.wrappedValue = RecurrentGemmaRecurrentBlock(config)
        } else {
            _temporalBlock.wrappedValue = RecurrentGemmaLocalAttentionBlock(config)
        }
        _channelPreNorm.wrappedValue = RecurrentGemmaRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
        _mlpBlock.wrappedValue = RecurrentGemmaMLPBlock(config)
    }

    var isAttention: Bool {
        temporalBlockType == "attention"
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let normalized = temporalPreNorm(input)
        let temporalOutput: MLXArray
        if isAttention {
            guard let attention = temporalBlock as? RecurrentGemmaLocalAttentionBlock else {
                preconditionFailure("RecurrentGemma attention block is missing attention")
            }
            temporalOutput = attention(normalized, mask: mask, cache: cache)
        } else {
            guard let recurrent = temporalBlock as? RecurrentGemmaRecurrentBlock else {
                preconditionFailure("RecurrentGemma recurrent block is missing recurrent state")
            }
            temporalOutput = recurrent(normalized, cache: cache as? MambaCache)
        }

        let residual = input + temporalOutput
        return residual + mlpBlock(channelPreNorm(residual))
    }
}

private final class RecurrentGemmaBackbone: Module {
    private let config: RecurrentGemmaConfiguration
    private let plan: RecurrentGemmaLayerPlan

    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [RecurrentGemmaResidualBlock]
    @ModuleInfo(key: "final_norm") private var finalNorm: RecurrentGemmaRMSNorm

    init(_ config: RecurrentGemmaConfiguration) {
        self.config = config
        self.plan = RecurrentGemmaLayerPlan(config)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            RecurrentGemmaResidualBlock(config, layerIndex: $0)
        }
        _finalNorm.wrappedValue = RecurrentGemmaRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedTokens(inputs)
        if config.scaleEmbeddingsBySqrtDim {
            hiddenStates = (hiddenStates * MLXArray(sqrt(Float(config.hiddenSize)), dtype: .float32))
                .asType(hiddenStates.dtype)
        }

        let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
            if let attentionIndex = plan.firstAttentionIndex {
                createAttentionMask(
                    h: hiddenStates,
                    cache: cache?[attentionIndex],
                    windowSize: config.attentionWindowSize
                )
            } else {
                .none
            }

        for (index, layer) in layers.enumerated() {
            hiddenStates = layer(
                hiddenStates,
                mask: layer.isAttention ? attentionMask : .none,
                cache: cache?[index]
            )
        }

        return finalNorm(hiddenStates)
    }
}

internal final class RecurrentGemmaModel: Module, LLMModel, GreedyTokenModel {
    internal let modelType: String
    internal let vocabularySize: Int

    private let config: RecurrentGemmaConfiguration
    private let plan: RecurrentGemmaLayerPlan
    @ModuleInfo(key: "model") private var model: RecurrentGemmaBackbone
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?

    init(_ config: RecurrentGemmaConfiguration) {
        self.config = config
        self.plan = RecurrentGemmaLayerPlan(config)
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        _model.wrappedValue = RecurrentGemmaBackbone(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        plan.blockTypes.map { blockType in
            blockType == "attention"
                ? RotatingKVCache(maxSize: config.attentionWindowSize) as KVCache
                : MambaCache() as KVCache
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

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        for (key, value) in sanitized where key.contains("conv_1d.weight") && value.dim(-1) != 1 {
            sanitized[key] = value.movedAxis(source: 2, destination: 1)
        }
        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
            sanitized["lm_head.scales"] = nil
            sanitized["lm_head.biases"] = nil
        }
        return sanitized.filter { key, _ in
            !key.contains(".rotary_emb.inv_freq")
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        model.layers.flatMap { layer -> LoRALinearLayers in
            let mlpTargets: LoRALinearLayers = [(layer.mlpBlock, ["up_proj", "gate_proj"])]
            if layer.isAttention,
                let attention = layer.temporalBlock as? RecurrentGemmaLocalAttentionBlock {
                return [(attention, ["q_proj", "v_proj"])] + mlpTargets
            }
            if let recurrent = layer.temporalBlock as? RecurrentGemmaRecurrentBlock {
                return [(recurrent, ["linear_x", "linear_y", "linear_out"])] + mlpTargets
            }
            return mlpTargets
        }
    }

    private func logits(from hiddenStates: MLXArray) -> MLXArray {
        var logits = if let lmHead {
            lmHead(hiddenStates)
        } else {
            model.embedTokens.asLinear(hiddenStates)
        }
        if config.logitsSoftCap > 0 {
            logits = tanh(logits / config.logitsSoftCap) * config.logitsSoftCap
        }
        return logits
    }
}
