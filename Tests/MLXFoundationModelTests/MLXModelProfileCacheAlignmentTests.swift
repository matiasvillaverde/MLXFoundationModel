import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile cache alignment")
struct MLXModelProfileCacheAlignmentTests {
    @Test("MiniMax M3 profile requires prefill-step prompt cache reuse")
    func miniMaxM3ProfileRequiresPrefillStepPromptCacheReuse() {
        let optimization = Self.miniMaxM3Profile.optimizationProfile

        #expect(optimization.promptCacheReuseAlignment == .prefillStep)
        #expect(optimization.implementedFeatures.contains(.prefillStepPromptCacheReuse))
    }

    @Test("language model applies profile-required prompt cache alignment")
    func languageModelAppliesProfileRequiredPromptCacheAlignment() {
        let runtime = ModelRuntimePreferences(
            promptCachePolicy: .memory,
            promptCacheByteLimit: 16_384,
            promptCacheReuseAlignment: .exact
        )
        let languageModel = MLXLanguageModel(
            model: MLXModel(
                id: "minimax-m3-fixture",
                location: URL(fileURLWithPath: "/tmp/minimax-m3-fixture"),
                profile: Self.miniMaxM3Profile
            ),
            runtime: runtime
        )

        #expect(languageModel.runtime.promptCacheReuseAlignment == .prefillStep)
        #expect(languageModel.runtime.promptCachePolicy == .memory)
        #expect(languageModel.runtime.promptCacheByteLimit == 16_384)
        #expect(languageModel.providerConfiguration.runtime?.promptCacheReuseAlignment == .prefillStep)
    }

    private static var miniMaxM3Profile: MLXModelProfile {
        MLXModelProfile.make(
            config: [
                "model_type": "minimax_m3",
                "architectures": ["MiniMaxM3ForCausalLM"]
            ]
        )
    }
}
