import Foundation

extension MLXOQQuantizationPlanner {
    /// Builds an oMLX-style budgeted quantization plan from tensor metadata.
    public func plan(
        for tensors: [MLXOQTensorDescriptor],
        options: MLXOQQuantizationPlanOptions = .init()
    ) -> MLXOQQuantizationPlan {
        let target = options.targetBitsPerWeight ?? level.defaultTargetBitsPerWeight
        let hardCap = options.hardCapBitsPerWeight ?? level.defaultHardCapBitsPerWeight
        var state = initialPlanState(for: tensors, fixedOverrides: options.fixedOverrides)
        let baselineBitsPerWeight = state.effectiveBitsPerWeight

        applyMandatoryBoosts(state: &state, hardCap: hardCap)
        applyFractionalExpertBoosts(state: &state)
        applyProtectionFloors(state: &state, hardCap: hardCap)
        applySensitivityBoosts(
            state: &state,
            hardCap: hardCap,
            layerSensitivityScores: options.layerSensitivityScores
        )
        applyFallbackBoosts(
            state: &state,
            target: target,
            hardCap: hardCap,
            layerSensitivityScores: options.layerSensitivityScores
        )

        return state.plan(
            level: level,
            baselineBitsPerWeight: baselineBitsPerWeight,
            targetBitsPerWeight: target,
            hardCapBitsPerWeight: hardCap
        )
    }

    private func initialPlanState(
        for tensors: [MLXOQTensorDescriptor],
        fixedOverrides: [String: MLXOQQuantizationSpec]
    ) -> MLXOQQuantizationPlanState {
        var state = MLXOQQuantizationPlanState(
            tensors: tensors,
            fixedOverrides: fixedOverrides
        )
        for tensor in tensors {
            guard budgetBaseSpec(for: tensor) != nil else {
                state.record(tensor: tensor, decision: .keepFullPrecision)
                continue
            }
            let spec = fixedOverrides[tensor.name] ?? baseSpec
            state.record(tensor: tensor, decision: .quantize(spec))
        }
        return state
    }

    private func budgetBaseSpec(
        for tensor: MLXOQTensorDescriptor
    ) -> MLXOQQuantizationSpec? {
        guard decision(for: tensor).quantizationSpec != nil else {
            return nil
        }
        return baseSpec
    }

    private var baseSpec: MLXOQQuantizationSpec {
        MLXOQQuantizationSpec(bits: level.baseBits, groupSize: defaultGroupSize)
    }
}

extension MLXOQLevel {
    var defaultTargetBitsPerWeight: Double? {
        switch value {
        case "2":
            2.8

        case "2.5":
            3.1

        case "2.7":
            3.35

        case "3":
            3.5

        case "3.5":
            3.8

        case "4":
            4.6

        case "5":
            5.5

        case "6":
            6.5

        default:
            nil
        }
    }

    var defaultHardCapBitsPerWeight: Double? {
        switch value {
        case "2":
            3.0

        case "2.5":
            3.3

        case "2.7":
            3.45

        case "3":
            3.7

        case "3.5":
            4.0

        case "4":
            4.7

        case "5":
            5.7

        case "6":
            6.7

        default:
            nil
        }
    }
}
