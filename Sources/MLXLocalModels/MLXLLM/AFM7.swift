import Foundation
import MLX
import MLXFast
import MLXNN

internal struct AFM7Configuration: Codable, Equatable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenDim: Int
    var layerCount: Int
    var kvReuseLayerCount: Int
    var attentionHeads: Int
    var kvHeads: Int
    var hiddenDimScaleFactor: Float
    var ropeTheta: Float
    var rmsNormEps: Float

    internal init(
        modelType: String = "afm7",
        vocabularySize: Int,
        hiddenDim: Int,
        layerCount: Int,
        kvReuseLayerCount: Int,
        attentionHeads: Int,
        kvHeads: Int,
        hiddenDimScaleFactor: Float = 3.25,
        ropeTheta: Float = 50_000,
        rmsNormEps: Float = 1e-5
    ) {
        precondition(layerCount > kvReuseLayerCount, "AFM7 requires at least one base layer")
        precondition(kvReuseLayerCount >= 0, "AFM7 KV reuse layer count cannot be negative")

        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenDim = hiddenDim
        self.layerCount = layerCount
        self.kvReuseLayerCount = kvReuseLayerCount
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.hiddenDimScaleFactor = hiddenDimScaleFactor
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
    }

    var baseLayerCount: Int {
        layerCount - kvReuseLayerCount
    }

    var feedForwardSize: Int {
        Int(Float(hiddenDim) * hiddenDimScaleFactor)
    }

    private enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenDim = "hidden_dim"
        case layerCount = "num_layers"
        case kvReuseLayerCount = "num_kv_reuse_layers"
        case attentionHeads = "num_heads"
        case kvHeads = "num_kv_heads"
        case hiddenDimScaleFactor = "hidden_dim_scale_factor"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
    }

    internal init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            modelType: try container.decodeIfPresent(String.self, forKey: .modelType) ?? "afm7",
            vocabularySize: try container.decode(Int.self, forKey: .vocabularySize),
            hiddenDim: try container.decode(Int.self, forKey: .hiddenDim),
            layerCount: try container.decode(Int.self, forKey: .layerCount),
            kvReuseLayerCount: try container.decode(Int.self, forKey: .kvReuseLayerCount),
            attentionHeads: try container.decode(Int.self, forKey: .attentionHeads),
            kvHeads: try container.decode(Int.self, forKey: .kvHeads),
            hiddenDimScaleFactor: try container.decodeIfPresent(
                Float.self,
                forKey: .hiddenDimScaleFactor
            ) ?? 3.25,
            ropeTheta: try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 50_000,
            rmsNormEps: try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        )
    }
}

internal struct AFM7AttentionLayout: Equatable, Sendable {
    let hiddenDim: Int
    let queryHeads: Int
    let keyValueHeads: Int
    let headDim: Int
    let queryProjectionSize: Int
    let keyValueProjectionSize: Int
    let combinedProjectionSize: Int
    let attentionScale: Float

    init(_ config: AFM7Configuration) {
        precondition(config.hiddenDim > 0, "AFM7 hidden dimension must be positive")
        precondition(config.attentionHeads > 0, "AFM7 attention head count must be positive")
        precondition(config.kvHeads > 0, "AFM7 KV head count must be positive")
        precondition(
            config.hiddenDim.isMultiple(of: config.attentionHeads),
            "AFM7 hidden dimension must divide evenly across attention heads"
        )
        precondition(
            config.attentionHeads.isMultiple(of: config.kvHeads),
            "AFM7 attention heads must divide evenly across KV heads"
        )

        self.hiddenDim = config.hiddenDim
        self.queryHeads = config.attentionHeads
        self.keyValueHeads = config.kvHeads
        self.headDim = config.hiddenDim / config.attentionHeads
        self.queryProjectionSize = queryHeads * headDim
        self.keyValueProjectionSize = keyValueHeads * headDim
        self.combinedProjectionSize = queryProjectionSize + 2 * keyValueProjectionSize
        self.attentionScale = pow(Float(headDim), -0.5)
    }
}

private struct AFM7AttentionResult {
    let hiddenStates: MLXArray
    let keys: MLXArray
    let values: MLXArray
}

private func afm7Fake8BitQuant(_ input: MLXArray, scale: MLXArray) -> MLXArray {
    let originalDType = input.dtype
    let quantized = MLX.round(input.asType(.float32) / scale.asType(.float32))
    let clipped = clip(quantized, min: MLXArray(Float(-128)), max: MLXArray(Float(127)))
    return (clipped * scale.asType(.float32)).asType(originalDType)
}

private final class AFM7Attention: Module {
    let layout: AFM7AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "qkv_proj") private var qkvProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") private var keyNorm: RMSNorm
    @ParameterInfo(key: "quant_key_scale") private var quantKeyScale: MLXArray
    @ParameterInfo(key: "quant_value_scale") private var quantValueScale: MLXArray

    init(_ config: AFM7Configuration) {
        self.layout = AFM7AttentionLayout(config)
        self.rope = initializeRope(
            dims: layout.headDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: nil,
            maxPositionEmbeddings: nil
        )
        _qkvProjection.wrappedValue = Linear(
            layout.hiddenDim,
            layout.combinedProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.hiddenDim,
            layout.hiddenDim,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim)
        _keyNorm.wrappedValue = RMSNorm(dimensions: layout.headDim)
        _quantKeyScale.wrappedValue = MLXArray(Float(1))
        _quantValueScale.wrappedValue = MLXArray(Float(1))
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> AFM7AttentionResult {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        let offset = cache?.offset ?? 0

        let projected = qkvProjection(input)
        let splitPoints = [
            layout.queryProjectionSize,
            layout.queryProjectionSize + layout.keyValueProjectionSize
        ]
        let qkv = split(projected, indices: splitPoints, axis: -1)

        let queries = queryNorm(
            rope(
                qkv[0]
                    .reshaped(batchSize, sequenceLength, layout.queryHeads, layout.headDim)
                    .transposed(0, 2, 1, 3),
                offset: offset
            )
        )
        let keys = afm7Fake8BitQuant(
            keyNorm(
                rope(
                    qkv[1]
                        .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
                        .transposed(0, 2, 1, 3),
                    offset: offset
                )
            ),
            scale: quantKeyScale
        )
        let values = afm7Fake8BitQuant(
            qkv[2]
                .reshaped(batchSize, sequenceLength, layout.keyValueHeads, layout.headDim)
                .transposed(0, 2, 1, 3),
            scale: quantValueScale
        )

        let result = attentionWithCacheUpdateReturningKV(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: layout.attentionScale,
            mask: mask
        )
        let output = result.output
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)

        return AFM7AttentionResult(
            hiddenStates: outputProjection(output),
            keys: result.keys,
            values: result.values
        )
    }
}

private final class AFM7KVReuseAttention: Module {
    private let layout: AFM7AttentionLayout
    private let rope: RoPELayer

    @ModuleInfo(key: "q_proj") private var queryProjection: Linear
    @ModuleInfo(key: "out_proj") private var outputProjection: Linear
    @ModuleInfo(key: "q_norm") private var queryNorm: RMSNorm

    init(_ config: AFM7Configuration) {
        self.layout = AFM7AttentionLayout(config)
        self.rope = initializeRope(
            dims: layout.headDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: nil,
            maxPositionEmbeddings: nil
        )
        _queryProjection.wrappedValue = Linear(
            layout.hiddenDim,
            layout.queryProjectionSize,
            bias: false
        )
        _outputProjection.wrappedValue = Linear(
            layout.hiddenDim,
            layout.hiddenDim,
            bias: false
        )
        _queryNorm.wrappedValue = RMSNorm(dimensions: layout.headDim)
    }

    func callAsFunction(
        _ input: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let batchSize = input.dim(0)
        let sequenceLength = input.dim(1)
        let keySequenceLength = keys.dim(-2)
        let offset = keySequenceLength - sequenceLength

        let queries = queryNorm(
            rope(
                queryProjection(input)
                    .reshaped(batchSize, sequenceLength, layout.queryHeads, layout.headDim)
                    .transposed(0, 2, 1, 3),
                offset: offset
            )
        )
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: layout.attentionScale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return outputProjection(output)
    }
}

private final class AFM7MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") private var gateProjection: Linear
    @ModuleInfo(key: "down_proj") private var downProjection: Linear
    @ModuleInfo(key: "up_proj") private var upProjection: Linear

    init(_ config: AFM7Configuration) {
        _gateProjection.wrappedValue = Linear(
            config.hiddenDim,
            config.feedForwardSize,
            bias: false
        )
        _downProjection.wrappedValue = Linear(
            config.feedForwardSize,
            config.hiddenDim,
            bias: false
        )
        _upProjection.wrappedValue = Linear(
            config.hiddenDim,
            config.feedForwardSize,
            bias: false
        )
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        downProjection(silu(gateProjection(input)) * upProjection(input))
    }
}

private final class AFM7DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: AFM7Attention
    @ModuleInfo(key: "mlp") fileprivate var feedForward: AFM7MLP
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: AFM7Configuration) {
        _attention.wrappedValue = AFM7Attention(config)
        _feedForward.wrappedValue = AFM7MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenDim,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenDim,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> AFM7AttentionResult {
        let attentionOutput = attention(inputLayerNorm(input), mask: mask, cache: cache)
        let attended = input + attentionOutput.hiddenStates
        let output = attended + feedForward(postAttentionLayerNorm(attended))
        return AFM7AttentionResult(
            hiddenStates: output,
            keys: attentionOutput.keys,
            values: attentionOutput.values
        )
    }
}

private final class AFM7KVReuseLayer: Module {
    @ModuleInfo(key: "self_attn") fileprivate var attention: AFM7KVReuseAttention
    @ModuleInfo(key: "mlp") fileprivate var feedForward: AFM7MLP
    @ModuleInfo(key: "input_layernorm") private var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") private var postAttentionLayerNorm: RMSNorm

    init(_ config: AFM7Configuration) {
        _attention.wrappedValue = AFM7KVReuseAttention(config)
        _feedForward.wrappedValue = AFM7MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenDim,
            eps: config.rmsNormEps
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenDim,
            eps: config.rmsNormEps
        )
    }

    func callAsFunction(
        _ input: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let attended = input + attention(
            inputLayerNorm(input),
            keys: keys,
            values: values,
            mask: mask
        )
        return attended + feedForward(postAttentionLayerNorm(attended))
    }
}

private final class AFM7Backbone: Module {
    @ModuleInfo(key: "embedding") fileprivate var embedding: Embedding
    @ModuleInfo(key: "layers") fileprivate var layers: [AFM7DecoderLayer]
    @ModuleInfo(key: "kv_reuse_layers") fileprivate var kvReuseLayers: [AFM7KVReuseLayer]
    @ModuleInfo(key: "output_norm") private var outputNorm: RMSNorm

    init(_ config: AFM7Configuration) {
        _embedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenDim
        )
        _layers.wrappedValue = (0 ..< config.baseLayerCount).map { _ in
            AFM7DecoderLayer(config)
        }
        _kvReuseLayers.wrappedValue = (0 ..< config.kvReuseLayerCount).map { _ in
            AFM7KVReuseLayer(config)
        }
        _outputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var hiddenStates = embedding(inputs)
        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)
        var sharedKV: (keys: MLXArray, values: MLXArray)?

        for (layerIndex, layer) in layers.enumerated() {
            let output = layer(hiddenStates, mask: mask, cache: cache?[layerIndex])
            hiddenStates = output.hiddenStates
            sharedKV = (keys: output.keys, values: output.values)
        }

        if !kvReuseLayers.isEmpty {
            guard let sharedKV else {
                preconditionFailure("AFM7 KV reuse layers require at least one base layer")
            }
            for layer in kvReuseLayers {
                hiddenStates = layer(
                    hiddenStates,
                    keys: sharedKV.keys,
                    values: sharedKV.values,
                    mask: mask
                )
            }
        }

        return outputNorm(hiddenStates)
    }
}

internal final class AFM7Model: Module, LLMModel, KVCacheDimensionProvider, GreedyTokenModel {
    internal let modelType: String
    internal let vocabularySize: Int
    internal let kvHeads: [Int]

    private let configuration: AFM7Configuration
    @ModuleInfo(key: "model") private var model: AFM7Backbone

    init(_ config: AFM7Configuration) {
        self.configuration = config
        self.modelType = config.modelType
        self.vocabularySize = config.vocabularySize
        self.kvHeads = Array(repeating: config.kvHeads, count: config.baseLayerCount)
        _model.wrappedValue = AFM7Backbone(config)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedding.asLinear(model(inputs, cache: cache))
    }

    internal func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        let hiddenStates = lastTokenHiddenState(model(input[text: .newAxis].tokens, cache: cache))
        return greedyTokenOutput(logits: model.embedding.asLinear(hiddenStates), state: state)
    }

    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.contains(".rotary_emb.inv_freq") && !Self.isOutputHead(key)
        }
    }

    internal func loraLinearLayers() -> LoRALinearLayers {
        let baseLayers: LoRALinearLayers = model.layers.map {
            ($0.attention, ["qkv_proj", "out_proj"])
        }
        let reuseLayers: LoRALinearLayers = model.kvReuseLayers.map {
            ($0.attention, ["q_proj", "out_proj"])
        }
        return baseLayers + reuseLayers
    }

    private static func isOutputHead(_ key: String) -> Bool {
        key == "lm_head.weight" || key == "lm_head.scales" || key == "lm_head.biases"
    }
}
