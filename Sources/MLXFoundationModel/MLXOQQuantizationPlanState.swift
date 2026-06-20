import Foundation

struct MLXOQQuantizationPlanState {
    private(set) var boosts: [String: MLXOQQuantizationSpec] = [:]
    private(set) var decisions: [String: MLXOQQuantizationDecision] = [:]
    let fixedOverrides: [String: MLXOQQuantizationSpec]
    let tensors: [String: MLXOQTensorDescriptor]

    init(
        tensors: [MLXOQTensorDescriptor],
        fixedOverrides: [String: MLXOQQuantizationSpec]
    ) {
        self.fixedOverrides = fixedOverrides
        self.tensors = Dictionary(uniqueKeysWithValues: tensors.map { tensor in
            (tensor.name, tensor)
        })
    }

    var effectiveBitsPerWeight: Double {
        guard quantizedParameterCount > 0 else {
            return 0
        }
        return Double(estimatedSerializedBytes * 8) / Double(quantizedParameterCount)
    }

    var estimatedSerializedBytes: Int {
        decisions.reduce(0) { partialResult, entry in
            guard let tensor = tensors[entry.key] else {
                return partialResult
            }
            return partialResult + estimatedBytes(for: tensor, decision: entry.value)
        }
    }

    var quantizedParameterCount: Int {
        decisions.reduce(0) { partialResult, entry in
            guard case .quantize = entry.value,
                let tensor = tensors[entry.key] else {
                return partialResult
            }
            return partialResult + parameterCount(for: tensor)
        }
    }

    var fullPrecisionParameterCount: Int {
        totalParameterCount - quantizedParameterCount
    }

    var totalParameterCount: Int {
        tensors.values.reduce(0) { partialResult, tensor in
            partialResult + parameterCount(for: tensor)
        }
    }

    mutating func record(
        tensor: MLXOQTensorDescriptor,
        decision: MLXOQQuantizationDecision
    ) {
        decisions[tensor.name] = decision
    }

    mutating func applyBoost(
        _ spec: MLXOQQuantizationSpec,
        to tensor: MLXOQTensorDescriptor,
        hardCapBitsPerWeight: Double?,
        ignoreHardCap: Bool = false
    ) -> Bool {
        guard !fixedOverrides.keys.contains(tensor.name),
            case .quantize(let currentSpec) = decisions[tensor.name],
            isHigherPrecision(spec, than: currentSpec) else {
            return false
        }
        let originalDecision = decisions[tensor.name]
        decisions[tensor.name] = .quantize(spec)
        guard ignoreHardCap || fitsHardCap(hardCapBitsPerWeight) else {
            decisions[tensor.name] = originalDecision
            return false
        }
        boosts[tensor.name] = spec
        return true
    }

    func currentSpec(for tensor: MLXOQTensorDescriptor) -> MLXOQQuantizationSpec? {
        decisions[tensor.name]?.quantizationSpec
    }

    func plan(
        level: MLXOQLevel,
        baselineBitsPerWeight: Double,
        targetBitsPerWeight: Double?,
        hardCapBitsPerWeight: Double?
    ) -> MLXOQQuantizationPlan {
        MLXOQQuantizationPlan(
            level: level,
            decisions: decisions,
            boosts: boosts,
            fixedOverrides: fixedOverrides,
            baselineBitsPerWeight: baselineBitsPerWeight,
            effectiveBitsPerWeight: effectiveBitsPerWeight,
            targetBitsPerWeight: targetBitsPerWeight,
            hardCapBitsPerWeight: hardCapBitsPerWeight,
            totalParameterCount: totalParameterCount,
            quantizedParameterCount: quantizedParameterCount,
            fullPrecisionParameterCount: fullPrecisionParameterCount,
            estimatedSerializedBytes: estimatedSerializedBytes
        )
    }

    private func fitsHardCap(_ hardCapBitsPerWeight: Double?) -> Bool {
        guard let hardCapBitsPerWeight else {
            return true
        }
        return effectiveBitsPerWeight <= hardCapBitsPerWeight
    }

    private func isHigherPrecision(
        _ candidate: MLXOQQuantizationSpec,
        than current: MLXOQQuantizationSpec
    ) -> Bool {
        candidate.bits > current.bits || candidate.groupSize < current.groupSize
    }

    private func estimatedBytes(
        for tensor: MLXOQTensorDescriptor,
        decision: MLXOQQuantizationDecision
    ) -> Int {
        switch decision {
        case .keepFullPrecision:
            parameterCount(for: tensor) * 2

        case .quantize(let spec):
            quantizedBytes(for: tensor, spec: spec)
        }
    }

    private func quantizedBytes(
        for tensor: MLXOQTensorDescriptor,
        spec: MLXOQQuantizationSpec
    ) -> Int {
        guard let width = tensor.shape.last,
            width > 0,
            width.isMultiple(of: spec.groupSize) else {
            return parameterCount(for: tensor) * 2
        }
        let parameters = parameterCount(for: tensor)
        let rows = parameters / width
        let groups = width / spec.groupSize
        let weightBytes = (parameters * spec.bits + 7) / 8
        let overheadBytes = rows * groups * bytesPerGroup(for: spec.mode)
        return weightBytes + overheadBytes
    }

    private func bytesPerGroup(for mode: String) -> Int {
        switch mode {
        case "mxfp4":
            1

        case "mxfp8":
            2

        default:
            4
        }
    }

    private func parameterCount(for tensor: MLXOQTensorDescriptor) -> Int {
        tensor.shape.reduce(1) { partialResult, dimension in
            partialResult * max(0, dimension)
        }
    }
}
