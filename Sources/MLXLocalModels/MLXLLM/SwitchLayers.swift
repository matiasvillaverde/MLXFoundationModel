import Foundation
import MLX
import MLXNN

internal struct SwitchExpertPermutation {
    internal let sortedInput: MLXArray
    internal let sortedExpertIndices: MLXArray
    internal let restoreOrder: MLXArray
    internal let outputPrefixShape: [Int]

    internal init(input: MLXArray, expertIndices: MLXArray) {
        let choicesPerToken = expertIndices.dim(-1)
        let flatExpertIndices = expertIndices.flattened()
        let sortOrder = argSort(flatExpertIndices)

        self.sortedInput = input.flattened(start: 0, end: -3)[
            sortOrder.floorDivide(choicesPerToken)
        ]
        self.sortedExpertIndices = flatExpertIndices[sortOrder]
        self.restoreOrder = argSort(sortOrder)
        self.outputPrefixShape = expertIndices.shape
    }

    internal func restore(_ output: MLXArray) -> MLXArray {
        unflatten(output[restoreOrder], axis: 0, shape: outputPrefixShape)
    }
}

internal enum SwitchExpertDispatch {
    internal static let sortedDispatchThreshold = 64

    internal static func shouldSort(expertIndices: MLXArray) -> Bool {
        expertIndices.size > sortedDispatchThreshold
    }
}

internal func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let permutation = SwitchExpertPermutation(input: x, expertIndices: indices)
    return (
        permutation.sortedInput,
        permutation.sortedExpertIndices,
        permutation.restoreOrder
    )
}

internal func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

internal final class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation

        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            bias: bias
        )
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            bias: bias
        )
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims,
            outputDims: inputDims,
            numExperts: numExperts,
            bias: bias
        )

        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        let routedInput = MLX.expandedDimensions(x, axes: [-2, -3])

        let sortedDispatch = SwitchExpertDispatch.shouldSort(expertIndices: indices)
        let permutation = sortedDispatch
            ? SwitchExpertPermutation(input: routedInput, expertIndices: indices)
            : nil

        let expertInput = permutation?.sortedInput ?? routedInput
        let expertIndices = permutation?.sortedExpertIndices ?? indices
        let expertOutput = downProj(
            activation(
                gateProj(expertInput, expertIndices, sortedIndices: sortedDispatch)
            )
                * upProj(expertInput, expertIndices, sortedIndices: sortedDispatch),
            expertIndices,
            sortedIndices: sortedDispatch
        )

        return MLX.squeezed(
            permutation?.restore(expertOutput) ?? expertOutput,
            axis: -2
        )
    }
}

internal class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        precondition(inputDims > 0, "inputDims must be positive")
        precondition(outputDims > 0, "outputDims must be positive")
        precondition(numExperts > 0, "numExperts must be positive")

        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    init(
        inputDims: Int,
        outputDims: Int,
        numExperts: Int,
        weight: MLXArray,
        bias: MLXArray? = nil
    ) {
        precondition(inputDims > 0, "inputDims must be positive")
        precondition(outputDims > 0, "outputDims must be positive")
        precondition(numExperts > 0, "numExperts must be positive")

        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let result = MLX.gatherMM(
            x,
            weight.swappedAxes(-1, -2),
            rhsIndices: indices,
            sortedIndices: sortedIndices
        )

        return addingExpertBias(to: result, expertIndices: indices)
    }

    func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) -> Module {
        QuantizedSwitchLinear(
            self,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    internal func addingExpertBias(to result: MLXArray, expertIndices: MLXArray) -> MLXArray {
        guard let bias else { return result }
        return result + MLX.expandedDimensions(bias[expertIndices], axis: -2)
    }
}

internal final class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(_ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        precondition(groupSize > 0, "groupSize must be positive")
        precondition(bits > 0, "bits must be positive")

        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims,
            outputDims: other.outputDims,
            numExperts: other.numExperts,
            weight: quantizedWeight,
            bias: other.bias
        )

        self.freeze()
    }

    override func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: self.mode,
            sortedIndices: sortedIndices
        )

        return addingExpertBias(to: result, expertIndices: indices)
    }
}
