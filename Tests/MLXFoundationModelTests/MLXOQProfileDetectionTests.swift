import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX oQ profile detection")
struct MLXOQProfileDetectionTests {
    struct IdentifierFixture: Sendable {
        let id: String
        let level: String
    }

    @Test(
        "detects oQ levels from model identifiers",
        arguments: [
            IdentifierFixture(id: "Qwen3.5-35B-A3B-oq4", level: "oQ4"),
            IdentifierFixture(id: "DeepSeek-V4-Flash-OQ3.5e", level: "oQ3.5e"),
            IdentifierFixture(id: "model-oq_2.7", level: "oQ2.7")
        ]
    )
    func detectsOQLevelsFromModelIdentifiers(_ fixture: IdentifierFixture) {
        let profile = MLXModelProfile.make(
            config: ["model_type": "qwen3_5"],
            id: fixture.id
        )
        let optimization = profile.optimizationProfile

        #expect(optimization.isOQQuantized)
        #expect(optimization.oQLevel == fixture.level)
        #expect(optimization.detectedFeatures.contains(.oQQuantization))
    }

    @Test("detects numeric oQ levels from quantization metadata")
    func detectsNumericOQLevelsFromQuantizationMetadata() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "deepseek_v4",
                "quantization_config": [
                    "bits": 4,
                    "group_size": 64,
                    "oq_level": 3.5,
                    "quant_method": "affine"
                ]
            ],
            id: "metadata-only"
        )
        let optimization = profile.optimizationProfile

        #expect(optimization.isOQQuantized)
        #expect(optimization.oQLevel == "oQ3.5")
        #expect(optimization.quantization?.bits == 4)
        #expect(optimization.quantization?.groupSize == 64)
    }

    @Test("detects string oQ levels from nested and embedded metadata")
    func detectsStringOQLevelsFromNestedAndEmbeddedMetadata() {
        let explicit = MLXModelProfile.make(
            config: [
                "model_type": "qwen3_5",
                "quantization_config": ["oQLevel": "oQ2.7"]
            ],
            id: nil
        )
        let embedded = MLXModelProfile.make(
            config: [
                "model_type": "qwen3_5",
                "quantization_config": ["quant_method": "omlx_oq8"]
            ],
            id: nil
        )

        #expect(explicit.optimizationProfile.oQLevel == "oQ2.7")
        #expect(embedded.optimizationProfile.oQLevel == "oQ8")
    }

    @Test("ignores unsupported oQ levels")
    func ignoresUnsupportedOQLevels() {
        let profile = MLXModelProfile.make(
            config: [
                "model_type": "qwen3_5",
                "quantization_config": ["oq_level": 7]
            ],
            id: "qwen-fixture"
        )
        let optimization = profile.optimizationProfile

        #expect(!optimization.isOQQuantized)
        #expect(optimization.oQLevel == nil)
        #expect(!optimization.detectedFeatures.contains(.oQQuantization))
    }
}
