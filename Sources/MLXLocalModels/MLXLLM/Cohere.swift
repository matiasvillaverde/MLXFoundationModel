import Foundation
import MLX

import MLXNN

// port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/cohere.py

private class Attention: Module {
    let args: CohereConfiguration
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: CohereConfiguration) {
        self.args = args

        let dim = args.hiddenSize
        let heads = args.attentionHeads
        let kvHeads = args.kvHeads

        let headDim = args.hiddenSize / heads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
        self._wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
        self._wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: args.ropeTraditional, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        // prepare the queries, keys and values for the attention computation
        queries = queries.reshaped(B, L, args.attentionHeads, -1).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, args.kvHeads, -1).transposed(0, 2, 1, 3)

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
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

private class MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        self._gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        self._down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

private class TransformerBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Attention
    let mlp: MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm

    init(_ args: CohereConfiguration) {
        self._attention.wrappedValue = Attention(args)
        self.mlp = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        self._inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = inputLayerNorm(x)
        let attnH = attention(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return attnH + ffH + x
    }
}

internal class CohereModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    fileprivate let layers: [TransformerBlock]
    let norm: LayerNorm

    public init(_ args: CohereConfiguration) {
        precondition(args.vocabularySize > 0)

        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        self.layers = (0 ..< args.hiddenLayers)
            .map { _ in
                TransformerBlock(args)
            }
        self.norm = LayerNorm(dimensions: args.hiddenSize, eps: args.layerNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let mask = createAttentionMask(h: h, cache: cache)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

internal class CohereModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let model: CohereModelInner
    let logitScale: Float

    public init(_ args: CohereConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = CohereModelInner(args)
        self.logitScale = args.logitScale
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var out = model(inputs, cache: cache)
        out = model.embedTokens.asLinear(out)
        out = out * self.logitScale
        return out
    }
}

internal struct CohereConfiguration: Codable, Sendable {
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var layerNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float = 8_000_000.0
    var ropeTraditional: Bool = true
    var ropeScaling: [String: StringOrNumber]?
    var logitScale: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        // custom implementation to handle optional keys with required values
        let container: KeyedDecodingContainer<Self.CodingKeys> =
            try decoder.container(
                keyedBy: Self.CodingKeys.self)

        self.hiddenSize = try container.decode(
            Int.self, forKey: Self.CodingKeys.hiddenSize)
        self.hiddenLayers = try container.decode(
            Int.self, forKey: Self.CodingKeys.hiddenLayers)
        self.intermediateSize = try container.decode(
            Int.self, forKey: Self.CodingKeys.intermediateSize)
        self.attentionHeads = try container.decode(
            Int.self, forKey: Self.CodingKeys.attentionHeads)
        self.layerNormEps = try container.decode(
            Float.self, forKey: Self.CodingKeys.layerNormEps)
        self.vocabularySize = try container.decode(
            Int.self, forKey: Self.CodingKeys.vocabularySize)
        self.kvHeads = try container.decode(
            Int.self, forKey: Self.CodingKeys.kvHeads)
        self.ropeTheta =
            try container.decodeIfPresent(
                Float.self, forKey: Self.CodingKeys.ropeTheta)
            ?? 8_000_000.0
        self.ropeScaling = try container.decodeIfPresent(
            [String: StringOrNumber].self, forKey: Self.CodingKeys.ropeScaling)
        self.logitScale = try container.decode(
            Float.self, forKey: Self.CodingKeys.logitScale)
    }
}

// MARK: - LoRA

extension CohereModel: LoRAModel {
    public func loraLinearLayers() -> LoRALinearLayers {
        model.layers.map { ($0.attention, ["q_proj", "v_proj"]) }
    }
}
