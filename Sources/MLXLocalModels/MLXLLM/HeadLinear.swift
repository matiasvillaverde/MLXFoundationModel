import MLX
import MLXNN

internal protocol HeadProjection {
    func project(_ input: MLXArray, transposedWeight: Bool) -> MLXArray
}

internal final class HeadLinear: Module, Quantizable, HeadProjection {
    @ParameterInfo(key: "weight") private var weight: MLXArray

    internal init(inputDims: Int, outputDims: Int, headCount: Int) {
        let scale = Float(1.0 / Double(inputDims)).squareRoot()
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [headCount, outputDims, inputDims]
        )
        super.init()
    }

    internal func project(_ input: MLXArray, transposedWeight: Bool = true) -> MLXArray {
        if transposedWeight {
            return input.matmul(weight.swappedAxes(-1, -2))
        }
        return input.matmul(weight)
    }

    internal func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedHeadLinear(
            weight: weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

internal final class QuantizedHeadLinear: Module, Quantized, HeadProjection {
    internal let groupSize: Int
    internal let bits: Int
    internal let mode: QuantizationMode

    @ParameterInfo(key: "weight") private var weight: MLXArray
    @ParameterInfo(key: "scales") private var scales: MLXArray
    @ParameterInfo(key: "biases") private var biases: MLXArray?

    internal init(
        weight: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self._weight.wrappedValue = quantizedWeight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    internal init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        self._weight.wrappedValue = weight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init()
        self.freeze()
    }

    internal func project(_ input: MLXArray, transposedWeight: Bool = true) -> MLXArray {
        quantizedMM(
            input,
            weight,
            scales: scales,
            biases: biases,
            transpose: transposedWeight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}

internal struct MLAKVProjectionSplitPlan: Equatable, Sendable {
    internal let headCount: Int
    internal let keyHeadDimensions: Int
    internal let valueHeadDimensions: Int
    internal let latentDimensions: Int

    internal init(
        headCount: Int,
        keyHeadDimensions: Int,
        valueHeadDimensions: Int,
        latentDimensions: Int
    ) {
        self.headCount = headCount
        self.keyHeadDimensions = keyHeadDimensions
        self.valueHeadDimensions = valueHeadDimensions
        self.latentDimensions = latentDimensions
    }

    internal var combinedHeadDimensions: Int {
        keyHeadDimensions + valueHeadDimensions
    }

    internal func split(weight: MLXArray) -> (embedQ: MLXArray, unembedOut: MLXArray) {
        let reshaped = weight.reshaped(headCount, combinedHeadDimensions, -1)
        let embedQ = contiguous(
            reshaped[0..., ..<keyHeadDimensions, 0...].swappedAxes(-1, -2)
        )
        let unembedOut = contiguous(reshaped[0..., keyHeadDimensions..., 0...])
        return (embedQ, unembedOut)
    }
}
