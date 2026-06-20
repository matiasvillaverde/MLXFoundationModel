import Foundation

enum MLXOQQuantizationRule: CaseIterable {
    case defaultProjectionFloors
    case fixedHighPrecision
    case fullPrecisionSmallOrSensitive
    case largeExpertProtection
    case projectionSensitivity
    case unquantizedFamilies

    static let evaluationOrder: [Self] = [
        .unquantizedFamilies,
        .fullPrecisionSmallOrSensitive,
        .fixedHighPrecision,
        .largeExpertProtection,
        .projectionSensitivity,
        .defaultProjectionFloors
    ]

    func decision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        switch self {
        case .defaultProjectionFloors:
            defaultProjectionFloorDecision(context)

        case .fixedHighPrecision:
            fixedHighPrecisionDecision(context)

        case .fullPrecisionSmallOrSensitive:
            fullPrecisionDecision(context)

        case .largeExpertProtection:
            largeExpertProtectionDecision(context)

        case .projectionSensitivity:
            projectionSensitivityDecision(context)

        case .unquantizedFamilies:
            unquantizedFamilyDecision(context)
        }
    }

    private func unquantizedFamilyDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        let name = context.lowercasedName
        if containsAny(name, [
            "visual.",
            "vision_",
            "patch_embed",
            "pos_embed",
            "image_newline",
            "multi_modal_projector",
            "visual.merger",
            "image_norm",
            "temporal_embed",
            "audio_tower"
        ]) {
            return .keepFullPrecision
        }
        return nil
    }

    private func fullPrecisionDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        let name = context.lowercasedName
        if isMoERouter(name) ||
            containsAny(name, ["ssm_alpha", "ssm_beta", "a_log", "time_decay", "time_faaaa"]) ||
            context.name.hasSuffix(".D") ||
            name.hasSuffix("dt_bias") {
            return .keepFullPrecision
        }
        return nil
    }

    private func fixedHighPrecisionDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        let name = context.lowercasedName
        if name.contains("shared_expert_gate"),
            !name.contains("gate_proj") {
            return .quantize(context.spec(bits: 8))
        }
        if name.contains("conv1d"),
            name.contains("linear_attn") {
            return .quantize(context.spec(bits: 8))
        }
        if name.contains("linear_attn.out_proj") {
            return .quantize(context.spec(bits: 5))
        }
        if containsAny(name, ["ssm_output", "ssm_out", "lora.2"]) {
            return .quantize(context.spec(bits: 8))
        }
        if containsAny(name, ["lm_head", "output.weight", "classifier"]) {
            return .quantize(context.spec(bits: 6))
        }
        if name.contains("cross_attn"),
            name.contains("o_proj") {
            return .quantize(context.spec(bits: 6))
        }
        if containsAny(name, ["kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj"]) {
            return .quantize(context.spec(bits: 6))
        }
        if name.contains("shared_expert"),
            !name.hasSuffix("shared_expert_gate.weight") {
            return .quantize(context.spec(bits: 8))
        }
        return nil
    }

    private func largeExpertProtectionDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        guard context.traits.numExperts >= 512,
            context.traits.hiddenSize >= 4_096,
            !context.lowercasedName.contains("shared_expert") else {
            return nil
        }
        if context.lowercasedName.contains("gate_proj") {
            return .quantize(context.spec(bits: 4))
        }
        if context.lowercasedName.contains("down_proj") {
            return .quantize(context.spec(bits: 3))
        }
        return nil
    }

    private func projectionSensitivityDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        let name = context.lowercasedName
        if containsAny(name, ["v_proj", "v_a_proj", "v_b_proj"]) {
            return context.isSensitiveLayer ? .quantize(context.spec(bits: 6)) : nil
        }
        if containsAny(name, ["q_proj", "k_proj", "qkv_proj", "in_proj_qkv", "attn_qkv"]),
            context.isSensitiveLayer {
            return .quantize(context.spec(bits: 5))
        }
        return nil
    }

    private func defaultProjectionFloorDecision(
        _ context: MLXOQQuantizationRuleContext
    ) -> MLXOQQuantizationDecision? {
        let name = context.lowercasedName
        if name.contains("o_proj"),
            !name.contains("shared_expert") {
            return context.traits.isMixtureOfExperts ? nil : .quantize(context.spec(bits: 5))
        }
        if containsAny(name, ["down_proj", "w2", "mlp.fc2", "wo"]) {
            if context.isRoutedExpert {
                if let boost = context.level.routedExpertDownProjectionBoost {
                    return .quantize(context.spec(bits: context.level.baseBits + boost))
                }
                return nil
            }
            return .quantize(context.spec(bits: context.isSensitiveLayer ? 6 : 5))
        }
        if containsAny(name, ["in_proj_z", "in_proj_a", "in_proj_b", "delta_net"]) {
            return .quantize(context.spec(bits: 5))
        }
        if containsAny(name, ["mixer.in_proj", "mixer.out_proj", "x_proj", "dt_proj"]) {
            return .quantize(context.spec(bits: 5))
        }
        return nil
    }

    private func isMoERouter(_ path: String) -> Bool {
        if path.hasSuffix("mlp.gate.weight") ||
            path.hasSuffix(".router.weight") ||
            path.hasSuffix(".router.layer.weight") {
            return true
        }
        if path.hasSuffix(".gate.weight"),
            !path.contains("gate_proj") {
            return true
        }
        return path.contains(".gate.") && !path.contains("gate_proj")
    }

    private func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }
}
