@testable import MLXFoundationModel
import Testing

@Suite("MLX model optimization DFlash profiles")
struct MLXModelOptimizationProfileDFlashTests {
    @Test("infers Gemma 4 VLM MTP and DFlash candidate paths")
    func infersGemma4VLMMTPAndDFlashCandidatePaths() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "vision_config": ["image_size": 896]
            ],
            id: "gemma-4-26b-it"
        )
        let optimization = profile.optimizationProfile

        #expect(optimization.supportsVLMMTP)
        #expect(optimization.supportsDFlash)
        #expect(optimization.detectedFeatures.contains(.vlmMTP))
        #expect(optimization.pendingRuntimeFeatures == [.dFlash, .vlmMTP])
        #expect(!optimization.implementedFeatures.contains(.vlmMTP))
        #expect(profile.capabilities.vision)
    }

    @Test("DFlash candidates follow top-level oMLX target model types")
    func dFlashCandidatesFollowTopLevelOMLXTargetModelTypes() {
        let qwen = MLXModelProfile.make(config: ["model_type": "qwen3_5"])
        let gemma4 = MLXModelProfile.make(config: ["model_type": "gemma4_text"])
        let deepSeek = MLXModelProfile.make(config: ["model_type": "deepseek_v4"])
        let nestedQwen = MLXModelProfile.make(config: [
            "model_type": "minimax_m3",
            "text_config": ["model_type": "qwen3_5"]
        ])

        #expect(qwen.optimizationProfile.supportsDFlash)
        #expect(gemma4.optimizationProfile.supportsDFlash)
        #expect(!deepSeek.optimizationProfile.supportsDFlash)
        #expect(!nestedQwen.optimizationProfile.supportsDFlash)
    }

    @Test("detects VLM MTP drafters without routing them through DFlash")
    func detectsVLMMTPDraftersWithoutRoutingThemThroughDFlash() {
        let profiles = [
            MLXModelProfile.make(config: [
                "model_type": "gemma4_assistant",
                "text_config": ["model_type": "gemma4_text"]
            ]),
            MLXModelProfile.make(config: ["model_type": "gemma4_unified_assistant"]),
            MLXModelProfile.make(config: ["model_type": "qwen3_5_mtp"])
        ]

        for profile in profiles {
            let optimization = profile.optimizationProfile

            #expect(optimization.supportsVLMMTPDrafter)
            #expect(!optimization.supportsVLMMTP)
            #expect(!optimization.supportsDFlash)
            #expect(optimization.detectedFeatures.contains(.vlmMTPDrafter))
            #expect(optimization.pendingRuntimeFeatures.contains(.vlmMTPDrafter))
            #expect(!optimization.implementedFeatures.contains(.vlmMTPDrafter))
        }
    }
}
