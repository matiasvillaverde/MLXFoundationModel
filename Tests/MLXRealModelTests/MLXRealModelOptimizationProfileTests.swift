import Foundation
@testable import MLXFoundationModel
import Testing

@Suite(
    "MLX real-model optimization profiles",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelOptimizationProfileTests {
    @Test("Qwen3.5 config-only MTP declaration does not advertise native MTP")
    func qwen35ConfigOnlyMTPDeclarationDoesNotAdvertiseNativeMTP() throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3.5", in: models) else {
            return
        }
        let profile = try MLXModelProfile.load(
            from: MLXRealModelEnvironment.modelURL(for: model),
            id: model.id
        )
        let optimization = profile.optimizationProfile

        #expect(profile.modelType == "qwen3_5")
        #expect(!optimization.hasNativeMTPWeights)
        #expect(!optimization.supportsNativeMTP)
        #expect(!optimization.nativeMTPRuntimeSupported)
        #expect(!optimization.detectedFeatures.contains(.nativeMTP))
        #expect(!optimization.implementedFeatures.contains(.nativeMTP))
    }
}
