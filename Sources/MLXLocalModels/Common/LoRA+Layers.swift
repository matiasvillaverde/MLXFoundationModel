import Foundation
import MLX
import MLXNN
import MLXRandom

private enum LoRAParameterKey {
    static let inputAdapter = "lora_a"
    static let outputAdapter = "lora_b"

    static let trainable: Set<String> = [
        inputAdapter,
        outputAdapter
    ]
}

private struct LoRAAdapterWeights {
    let input: MLXArray
    let output: MLXArray
}

private func makeLoRAAdapterWeights(
    inputDimensions: Int,
    outputDimensions: Int,
    rank: Int
) -> LoRAAdapterWeights {
    precondition(inputDimensions > 0, "LoRA input dimension must be positive")
    precondition(outputDimensions > 0, "LoRA output dimension must be positive")
    precondition(rank > 0, "LoRA rank must be positive")

    let bound = 1 / sqrt(Float(inputDimensions))
    return LoRAAdapterWeights(
        input: MLXRandom.uniform(
            low: -bound,
            high: bound,
            [inputDimensions, rank]
        ),
        output: MLXArray.zeros([rank, outputDimensions])
    )
}

private func freezeKeys(
    for module: Module,
    requestedKeys: [String]?
) -> [String] {
    let localKeys =
        requestedKeys
        ?? module.filterMap(filter: type(of: module).filterLocalParameters)
            .flattened()
            .map { $0.0 }
    return localKeys.filter { !LoRAParameterKey.trainable.contains($0) }
}

private func lowRankWeightDelta(
    inputAdapter: MLXArray,
    outputAdapter: MLXArray,
    scale: Float,
    dtype: DType
) -> MLXArray {
    let output = (scale * outputAdapter.T).asType(dtype)
    let input = inputAdapter.T.asType(dtype)
    return matmul(output, input)
}

private func lowRankActivationDelta(
    input: MLXArray,
    inputAdapter: MLXArray,
    outputAdapter: MLXArray,
    scale: Float,
    dtype: DType
) -> MLXArray {
    let delta = matmul(matmul(input, inputAdapter), outputAdapter)
    return (scale * delta).asType(dtype)
}

/// Linear layer with a trainable LoRA adapter over frozen base weights.
internal class LoRALinear: Linear, LoRALayer {
    let scale: Float

    @ParameterInfo(key: LoRAParameterKey.inputAdapter) var loraA: MLXArray
    @ParameterInfo(key: LoRAParameterKey.outputAdapter) var loraB: MLXArray

    required public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        rank: Int = 8,
        bias: Bool = false,
        scale: Float = 20.0,
        linear: Linear
    ) {
        self.scale = scale
        let adapter = makeLoRAAdapterWeights(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            rank: rank
        )
        _loraA.wrappedValue = adapter.input
        _loraB.wrappedValue = adapter.output

        super.init(weight: linear.weight, bias: linear.bias)
        freeze()
    }

    internal static func from(
        linear: Linear,
        rank: Int = 8,
        scale: Float = 20.0
    ) -> LoRALayer {
        if let quantized = linear as? QuantizedLinear {
            return QLoRALinear.from(linear: quantized, rank: rank, scale: scale)
        }

        let (outputDimensions, inputDimensions) = linear.shape
        return LoRALinear(
            inputDimensions,
            outputDimensions,
            rank: rank,
            scale: scale,
            linear: linear
        )
    }

    public override func freeze(
        recursive: Bool = true,
        keys: [String]? = nil,
        strict: Bool = false
    ) throws {
        try super.freeze(
            recursive: recursive,
            keys: freezeKeys(for: self, requestedKeys: keys),
            strict: strict
        )
    }

    internal func fused() -> Module {
        Linear(
            weight: weight + lowRankWeightDelta(
                inputAdapter: loraA,
                outputAdapter: loraB,
                scale: scale,
                dtype: weight.dtype
            ),
            bias: bias
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let base = super.callAsFunction(x.asType(weight.dtype))
        return base + lowRankActivationDelta(
            input: x,
            inputAdapter: loraA,
            outputAdapter: loraB,
            scale: scale,
            dtype: base.dtype
        )
    }
}

/// Quantized linear layer with a trainable LoRA adapter over frozen base weights.
internal class QLoRALinear: QuantizedLinear, LoRALayer {
    let scale: Float

    @ParameterInfo(key: LoRAParameterKey.inputAdapter) var loraA: MLXArray
    @ParameterInfo(key: LoRAParameterKey.outputAdapter) var loraB: MLXArray

    required public init(
        _ inputDimensions: Int,
        _ outputDimensions: Int,
        rank: Int = 8,
        bias: Bool = false,
        scale: Float = 20.0,
        linear: QuantizedLinear
    ) {
        self.scale = scale
        let adapter = makeLoRAAdapterWeights(
            inputDimensions: inputDimensions,
            outputDimensions: outputDimensions,
            rank: rank
        )
        _loraA.wrappedValue = adapter.input
        _loraB.wrappedValue = adapter.output

        super.init(
            weight: linear.weight,
            bias: linear.bias,
            scales: linear.scales,
            biases: linear.biases,
            groupSize: linear.groupSize,
            bits: linear.bits,
            mode: linear.mode
        )
        freeze()
    }

    internal static func from(
        linear: QuantizedLinear,
        rank: Int = 8,
        scale: Float = 20.0
    ) -> LoRALayer {
        let (outputDimensions, inputDimensions) = linear.shape
        return QLoRALinear(
            inputDimensions,
            outputDimensions,
            rank: rank,
            scale: scale,
            linear: linear
        )
    }

    public override func freeze(
        recursive: Bool = true,
        keys: [String]? = nil,
        strict: Bool = false
    ) throws {
        try super.freeze(
            recursive: recursive,
            keys: freezeKeys(for: self, requestedKeys: keys),
            strict: strict
        )
    }

    internal func fused() -> Module {
        let baseWeight = dequantizedWeight
        return QuantizedLinear(
            weight: baseWeight + lowRankWeightDelta(
                inputAdapter: loraA,
                outputAdapter: loraB,
                scale: scale,
                dtype: baseWeight.dtype
            ),
            bias: bias,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let base = super.callAsFunction(x.asType(scales.dtype))
        return base + lowRankActivationDelta(
            input: x,
            inputAdapter: loraA,
            outputAdapter: loraB,
            scale: scale,
            dtype: base.dtype
        )
    }
}
