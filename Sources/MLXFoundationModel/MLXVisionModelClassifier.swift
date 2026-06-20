import Foundation

enum MLXVisionModelClassifier {
    private static let visionEvidenceKeys = [
        "vision_config",
        "vit_config",
        "vision_tower",
        "mm_vision_tower",
        "image_token_index"
    ]

    private static let alwaysVisionModelTypes: Set<String> = [
        "diffusion_gemma",
        "gemma4_unified",
        "minimax_m3_vl"
    ]

    private static let vlmRuntimeModelTypes: Set<String> = [
        "cohere2_moe",
        "gemma4_unified",
        "minimax_m3",
        "minimax_m3_vl"
    ]

    private static let visionModelTypes: Set<String> = [
        "bunny_llama",
        "deepseekocr",
        "deepseekocr_2",
        "dots_ocr",
        "florence2",
        "gemma3",
        "gemma4",
        "glm_ocr",
        "idefics3",
        "internvl_chat",
        "llava",
        "llava_next",
        "llava_qwen2",
        "mistral3",
        "mllama",
        "molmo",
        "molmo2",
        "multi_modality",
        "paligemma",
        "phi3_v",
        "phi4_siglip",
        "phi4mm",
        "pixtral",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_5_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "youtu_vl"
    ]

    private static let visionArchitectures: Set<String> = [
        "Florence2ForConditionalGeneration",
        "Idefics3ForConditionalGeneration",
        "InternVLChatModel",
        "LlavaForConditionalGeneration",
        "LlavaNextForConditionalGeneration",
        "LlavaQwen2ForCausalLM",
        "MllamaForConditionalGeneration",
        "Molmo2ForConditionalGeneration",
        "MolmoForCausalLM",
        "PaliGemmaForConditionalGeneration",
        "Phi3VForCausalLM",
        "Pixtral",
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
        "MiniMaxM3VLForConditionalGeneration"
    ]

    private static let vlmRuntimeArchitectures: Set<String> = [
        "Cohere2MoeForCausalLM",
        "Gemma4UnifiedForConditionalGeneration",
        "MiniMaxM3ForCausalLM",
        "MiniMaxM3VLForConditionalGeneration"
    ]

    private static let evidenceRequiredVisionArchitectures: Set<String> = [
        "Gemma3ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration"
    ]

    static func hasVisionEvidence(in config: [String: Any]) -> Bool {
        visionEvidenceKeys.contains { key in
            if key == "mm_vision_tower", let value = config[key] as? String {
                return !value.isEmpty
            }
            return config[key] != nil
        }
    }

    static func isVisionModel(
        modelType: String?,
        architectures: [String],
        hasVisionEvidence: Bool
    ) -> Bool {
        let normalizedType = normalizeModelType(modelType)
        if let normalizedType, alwaysVisionModelTypes.contains(normalizedType) {
            return true
        }
        if hasVisionEvidence {
            return true
        }
        if architectures.contains(where: evidenceRequiredVisionArchitectures.contains) {
            return false
        }
        if let normalizedType, visionModelTypes.contains(normalizedType) {
            return false
        }
        return architectures.contains(where: visionArchitectures.contains)
    }

    static func runtimeKind(
        modelType: String?,
        architectures: [String],
        isVisionModel: Bool
    ) -> MLXModelRuntimeKind {
        let normalizedType = normalizeModelType(modelType)
        if isVisionModel ||
            normalizedType.map(vlmRuntimeModelTypes.contains) == true ||
            architectures.contains(where: vlmRuntimeArchitectures.contains) {
            return .vlm
        }
        return .text
    }

    private static func normalizeModelType(_ modelType: String?) -> String? {
        modelType?
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
    }
}
