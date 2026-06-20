import Foundation
import MLXLocalModels

enum MLXModelOptimizationProfileFactory {
    struct Input {
        let id: String?
        let modelType: String?
        let architectures: [String]
        let config: [String: Any]
        let isMixtureOfExperts: Bool
        let isVisionModel: Bool
        let requiresVLMRuntime: Bool
        let hasNativeMTPWeightTensors: Bool
        let hasFP8ScaleSidecars: Bool
    }

    private static let nativeMTPNeedles = [
        "qwen3_5",
        "qwen3.5",
        "qwen3_6",
        "qwen3.6",
        "deepseek_v4",
        "deepseek-v4"
    ]

    private static let nativeMTPRuntimeNeedles = [
        "qwen3_5",
        "qwen3.5"
    ]

    private static let vlmMTPNeedles = [
        "gemma4",
        "gemma-4",
        "qwen3_5",
        "qwen3.5",
        "qwen3_6",
        "qwen3.6"
    ]

    private static let vlmMTPDrafterNeedles = [
        "gemma4_assistant",
        "gemma4-assistant",
        "gemma4_unified_assistant",
        "gemma4-unified-assistant",
        "qwen3_5_mtp",
        "qwen3.5_mtp",
        "qwen3.5-mtp"
    ]

    private static let dFlashExcludedModelTypes: Set<String> = [
        "gemma4_assistant",
        "gemma4_unified_assistant",
        "qwen3_5_mtp"
    ]

    private static let prefillStepCacheReuseNeedles = [
        "minimax_m3",
        "minimax-m3",
        "minimaxm3"
    ]

    private static let deepSeekV32Needles = [
        "deepseek_v32",
        "deepseek-v32",
        "deepseekv32",
        "deepseek_v3.2",
        "deepseek-v3.2"
    ]

    static func make(input: Input) -> MLXModelOptimizationProfile {
        let text = searchableText(input)
        let quantizationConfig = quantizationConfig(input.config)
        let quantization = quantizationProfile(quantizationConfig)
        let oQLevel = MLXOQLevelParser.detect(
            id: input.id,
            config: input.config,
            quantizationConfig: quantizationConfig
        )
        let hasNativeMTPWeights = input.hasNativeMTPWeightTensors
        let supportsNativeMTP = hasNativeMTPWeights && containsAny(text, nativeMTPNeedles)
        let supportsVLMMTP = input.isVisionModel && containsAny(text, vlmMTPNeedles)
        let requiresFP8Dequantization = requiresFP8ScaleDequantization(input, quantization)

        return MLXModelOptimizationProfile(
            quantization: quantization,
            isOQQuantized: oQLevel != nil,
            oQLevel: oQLevel,
            requiresFP8ScaleDequantization: requiresFP8Dequantization,
            hasNativeMTPWeights: hasNativeMTPWeights,
            supportsNativeMTP: supportsNativeMTP,
            nativeMTPRuntimeSupported: nativeMTPRuntimeSupported(supportsNativeMTP, text),
            supportsVLMMTP: supportsVLMMTP,
            supportsVLMMTPDrafter: containsAny(text, vlmMTPDrafterNeedles),
            supportsSpeculativePrefill: input.isMixtureOfExperts,
            supportsDFlash: supportsDFlash(modelType: input.modelType),
            supportsIndexCache: detectsIndexCache(config: input.config, text: text),
            supportsTurboQuantKV: supportsTurboQuantKV(text: text),
            promptCacheReuseAlignment: promptCacheReuseAlignment(text: text),
            defaultIndexCacheFrequency: defaultIndexCacheFrequency(text: text)
        )
    }

    private static func quantizationProfile(_ nested: [String: Any])
        -> MLXModelQuantizationProfile? {
        let profile = MLXModelQuantizationProfile(
            bits: int(nested, keys: ["bits", "nbits"]),
            groupSize: int(nested, keys: ["group_size", "groupSize"]),
            method: string(nested, keys: ["quant_method", "quantization_method", "method"]),
            linearClass: string(nested, keys: ["linear_class", "linearClass"]),
            mode: string(nested, keys: ["quantization_mode", "mode"]),
            format: string(nested, keys: ["format", "dtype"])
        )
        return hasQuantizationMetadata(profile) ? profile : nil
    }

    private static func nativeMTPRuntimeSupported(
        _ supportsNativeMTP: Bool,
        _ text: String
    ) -> Bool {
        supportsNativeMTP && containsAny(text, nativeMTPRuntimeNeedles)
    }

    private static func quantizationConfig(_ config: [String: Any]) -> [String: Any] {
        for key in ["quantization", "quantization_config"] {
            if let nested = config[key] as? [String: Any] {
                return nested
            }
        }
        return config
    }

    private static func hasQuantizationMetadata(_ profile: MLXModelQuantizationProfile) -> Bool {
        profile.bits != nil
            || profile.groupSize != nil
            || profile.method != nil
            || profile.linearClass != nil
            || profile.mode != nil
            || profile.format != nil
    }

    private static func requiresFP8ScaleDequantization(
        _ input: Input,
        _ quantization: MLXModelQuantizationProfile?
    ) -> Bool {
        requiresFP8ScaleDequantization(
            quantization,
            hasFP8ScaleSidecars: input.hasFP8ScaleSidecars
        )
    }

    private static func requiresFP8ScaleDequantization(
        _ quantization: MLXModelQuantizationProfile?,
        hasFP8ScaleSidecars: Bool
    ) -> Bool {
        if hasFP8ScaleSidecars {
            return true
        }
        guard let quantization else {
            return false
        }
        let text = [
            quantization.method,
            quantization.linearClass,
            quantization.mode,
            quantization.format
        ]
        .compactMap(\.self)
        .joined(separator: "\n")
        .lowercased()
        return containsAny(text, ["fp8", "mxfp8", "float8"])
    }

    private static func detectsNativeMTPHeads(_ config: [String: Any]) -> Bool {
        if detectsNativeMTPHeads(in: config) {
            return true
        }
        if let textConfig = config["text_config"] as? [String: Any] {
            return detectsNativeMTPHeads(in: textConfig)
        }
        return false
    }

    private static func detectsNativeMTPHeads(in config: [String: Any]) -> Bool {
        int(config, keys: [
            "mtp_num_hidden_layers",
            "num_nextn_predict_layers",
            "num_next_predict_layers",
            "num_mtp_layers",
            "mtp_num_layers"
        ]) != nil
    }

    private static func detectsIndexCache(config: [String: Any], text: String) -> Bool {
        containsAnyKey(config, keys: ["kv_lora_rank", "qk_rope_head_dim"])
            || containsAny(text, ["deepseek", "glm4_moe"])
            || containsAny(text, prefillStepCacheReuseNeedles)
    }

    private static func supportsTurboQuantKV(text: String) -> Bool {
        !containsAny(text, prefillStepCacheReuseNeedles)
    }

    private static func promptCacheReuseAlignment(text: String) -> PromptCacheReuseAlignment? {
        containsAny(text, prefillStepCacheReuseNeedles) ? .prefillStep : nil
    }

    private static func defaultIndexCacheFrequency(text: String) -> Int? {
        containsAny(text, deepSeekV32Needles) ? 2 : nil
    }

    private static func supportsDFlash(modelType: String?) -> Bool {
        let type = normalizedModelType(modelType)
        guard !dFlashExcludedModelTypes.contains(type) else {
            return false
        }
        return type.hasPrefix("qwen") || type == "gemma4" || type == "gemma4_text"
    }

    private static func normalizedModelType(_ value: String?) -> String {
        value?
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: ".", with: "_") ?? ""
    }

    private static func searchableText(_ input: Input) -> String {
        ([input.modelType] + input.architectures + [input.id])
            .compactMap(\.self)
            .joined(separator: "\n")
            .lowercased()
    }

    private static func containsAnyKey(_ config: [String: Any], keys: [String]) -> Bool {
        keys.contains { config[$0] != nil }
    }

    private static func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }

    private static func string(_ config: [String: Any], keys: [String]) -> String? {
        for key in keys where config[key] is String {
            return config[key] as? String
        }
        return nil
    }

    private static func int(_ config: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = config[key] as? Int, value > 0 {
                return value
            }
            if let value = config[key] as? Double, value > 0 {
                return Int(value)
            }
            if let value = config[key] as? String, let parsed = Int(value), parsed > 0 {
                return parsed
            }
        }
        return nil
    }
}
