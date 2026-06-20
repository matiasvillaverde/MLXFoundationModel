import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX model pool visible models")
struct MLXModelPoolVisibleModelTests {
    @Test("serving profile catalog preserves VLM capabilities")
    func servingProfileCatalogPreservesVLMCapabilities() async throws {
        let pool = MLXModelPool()
        try await Self.registerVLMProfileFixture(in: pool)

        let visibleProfile = try await Self.visibleModel("gemma4:thinking", in: pool)

        Self.expectVLMThinkingProfile(visibleProfile)
    }

    private static func registerVLMProfileFixture(in pool: MLXModelPool) async throws {
        try await pool.register(
            Self.model("gemma4", profile: Self.gemma4VLMProfile),
            aliases: ["vision-default"],
            profiles: [Self.thinkingServingProfile]
        )
    }

    private static func visibleModel(
        _ id: String,
        in pool: MLXModelPool
    ) async throws -> MLXModelPoolVisibleModel {
        let snapshot = await pool.snapshot()
        return try #require(snapshot.visibleModels.first { model in
            model.id == id
        })
    }

    private static func expectVLMThinkingProfile(
        _ visibleProfile: MLXModelPoolVisibleModel
    ) {
        #expect(visibleProfile.sourceModelID == "gemma4")
        #expect(visibleProfile.aliases == ["vision-thinking"])
        #expect(visibleProfile.promptStyle == .gemma)
        #expect(visibleProfile.runtimeKind == .vlm)
        #expect(visibleProfile.capabilities.vision)
        #expect(visibleProfile.capabilities.reasoning)
        #expect(visibleProfile.capabilities.structuredOutput)
        #expect(visibleProfile.contextLength == 131_072)
        #expect(visibleProfile.maximumResponseTokens == 1_024)
    }

    private static var gemma4VLMProfile: MLXModelProfile {
        MLXModelProfile(
            contextLength: 131_072,
            promptStyle: .gemma,
            runtimeKind: .vlm,
            capabilities: MLXModelCapabilities(
                toolCalling: true,
                structuredOutput: true,
                vision: true,
                reasoning: true
            )
        )
    }

    private static var thinkingServingProfile: MLXModelServingProfile {
        MLXModelServingProfile(
            name: "thinking",
            aliases: ["vision-thinking"],
            maximumResponseTokens: 1_024
        )
    }

    private static func model(
        _ id: String,
        profile: MLXModelProfile
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: id,
                location: URL(fileURLWithPath: "/tmp/mlx-model-pool-visible-tests/\(id)"),
                profile: profile
            )
        )
    }
}
