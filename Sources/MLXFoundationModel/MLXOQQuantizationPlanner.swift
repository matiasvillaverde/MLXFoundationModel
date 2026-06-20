import Foundation

/// Swift implementation of oMLX's Universal Dynamic Quantization predicate.
public struct MLXOQQuantizationPlanner: Sendable {
    public let level: MLXOQLevel
    public let traits: MLXOQModelQuantizationTraits
    public let defaultGroupSize: Int

    public init(
        level: MLXOQLevel,
        traits: MLXOQModelQuantizationTraits = .init(),
        defaultGroupSize: Int = 64
    ) {
        self.level = level
        self.traits = traits
        self.defaultGroupSize = max(1, defaultGroupSize)
    }

    public init?(
        level: String,
        traits: MLXOQModelQuantizationTraits = .init(),
        defaultGroupSize: Int = 64
    ) {
        guard let parsed = MLXOQLevel(level) else {
            return nil
        }
        self.init(level: parsed, traits: traits, defaultGroupSize: defaultGroupSize)
    }

    public func decision(for tensor: MLXOQTensorDescriptor) -> MLXOQQuantizationDecision {
        guard Self.shouldQuantizeTensor(tensor) else {
            return .keepFullPrecision
        }
        let context = MLXOQQuantizationRuleContext(
            tensor: tensor,
            level: level,
            traits: traits,
            defaultGroupSize: defaultGroupSize
        )
        guard !context.isMTPProtected else {
            return .keepFullPrecision
        }
        for rule in MLXOQQuantizationRule.evaluationOrder {
            if let decision = rule.decision(context) {
                return normalized(decision, tensor: tensor)
            }
        }
        return normalized(.quantize(context.spec(bits: level.baseBits)), tensor: tensor)
    }

    public func decisions(
        for tensors: [MLXOQTensorDescriptor]
    ) -> [String: MLXOQQuantizationDecision] {
        Dictionary(uniqueKeysWithValues: tensors.map { tensor in
            (tensor.name, decision(for: tensor))
        })
    }

    public static func isQuantizableModel(config: [String: Any]) -> Bool {
        if config["quantization"] != nil {
            return false
        }
        guard let quantizationConfig = config["quantization_config"] as? [String: Any] else {
            return true
        }
        return quantizationConfig["quant_method"] as? String == "fp8"
    }

    private func normalized(
        _ decision: MLXOQQuantizationDecision,
        tensor: MLXOQTensorDescriptor
    ) -> MLXOQQuantizationDecision {
        guard case .quantize(let spec) = decision else {
            return decision
        }
        guard spec.bits > 0,
            spec.groupSize > 0,
            let width = tensor.shape.last,
            width.isMultiple(of: spec.groupSize) else {
            return .keepFullPrecision
        }
        return .quantize(spec)
    }

    private static func shouldQuantizeTensor(_ tensor: MLXOQTensorDescriptor) -> Bool {
        guard tensor.name.hasSuffix(".weight"),
            tensor.shape.count >= 2 else {
            return false
        }
        let lowercased = tensor.name.lowercased()
        guard !skipQuantPatterns.contains(where: lowercased.contains),
            !tensor.name.hasSuffix(".bias") else {
            return false
        }
        return true
    }

    private static let skipQuantPatterns = [
        "layernorm",
        "rmsnorm",
        "norm.weight",
        "norm.bias",
        "ln_",
        "layer_norm"
    ]
}
