import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX vision model profiles")
struct MLXVisionModelProfileTests {
    @Test("requires vision evidence for text-only Gemma and Qwen variants")
    func requiresVisionEvidenceForTextOnlyVariants() {
        let gemma = MLXModelProfile.make(config: [
            "model_type": "gemma4",
            "architectures": ["Gemma4ForConditionalGeneration"]
        ])
        let qwen = MLXModelProfile.make(config: [
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"]
        ])

        #expect(!gemma.hasVisionConfig)
        #expect(!gemma.isVisionModel)
        #expect(!gemma.capabilities.vision)
        #expect(gemma.optimizationProfile.supportsDFlash)
        #expect(!qwen.isVisionModel)
        #expect(qwen.optimizationProfile.supportsDFlash)
    }

    @Test("detects always VLM native model types without vision config")
    func detectsAlwaysVLMNativeTypesWithoutVisionConfig() {
        let profile = MLXModelProfile.make(config: [
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"]
        ])

        #expect(!profile.hasVisionConfig)
        #expect(profile.isVisionModel)
        #expect(profile.capabilities.vision)
        #expect(!profile.optimizationProfile.supportsDFlash)
        #expect(profile.optimizationProfile.supportsVLMMTP)
    }

    @Test("marks oMLX VLM-runtime text models without vision capability")
    func marksOMLXVLMRuntimeTextModelsWithoutVisionCapability() {
        let cohere = MLXModelProfile.make(config: [
            "model_type": "cohere2_moe",
            "architectures": ["Cohere2MoeForCausalLM"]
        ])
        let miniMax = MLXModelProfile.make(config: [
            "model_type": "minimax_m3",
            "architectures": ["MiniMaxM3ForCausalLM"]
        ])

        #expect(cohere.runtimeKind == .vlm)
        #expect(cohere.requiresVLMRuntime)
        #expect(!cohere.isVisionModel)
        #expect(!cohere.capabilities.vision)
        #expect(!cohere.optimizationProfile.supportsDFlash)

        #expect(miniMax.runtimeKind == .vlm)
        #expect(miniMax.promptStyle == .minimaxM3)
        #expect(!miniMax.isVisionModel)
        #expect(!miniMax.capabilities.vision)
        #expect(!miniMax.optimizationProfile.supportsDFlash)
    }

    @Test("marks MiniMax M3 VL as VLM runtime with vision capability")
    func marksMiniMaxM3VLAsVLMRuntimeWithVisionCapability() {
        let profile = MLXModelProfile.make(config: [
            "model_type": "minimax_m3_vl",
            "architectures": ["MiniMaxM3VLForConditionalGeneration"],
            "vision_config": ["hidden_size": 1_280],
            "text_config": ["hidden_size": 6_144]
        ])

        #expect(profile.runtimeKind == .vlm)
        #expect(profile.requiresVLMRuntime)
        #expect(profile.isVisionModel)
        #expect(profile.capabilities.vision)
        #expect(profile.promptStyle == .minimaxM3)
        #expect(!profile.optimizationProfile.supportsDFlash)
    }

    @Test("detects unknown VLM architectures")
    func detectsUnknownVLMArchitectures() {
        let profile = MLXModelProfile.make(config: [
            "model_type": "unknown_vlm",
            "architectures": ["LlavaForConditionalGeneration"]
        ])

        #expect(profile.runtimeKind == .vlm)
        #expect(profile.isVisionModel)
        #expect(profile.capabilities.vision)
        #expect(!profile.optimizationProfile.supportsDFlash)
    }
}
