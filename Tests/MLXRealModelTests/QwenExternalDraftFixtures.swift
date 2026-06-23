import Foundation
@testable import MLXLocalModels

enum QwenExternalDraftFixtures {
    struct Setup {
        let target: MLXRealModelCatalog.Model
        let draft: MLXRealModelCatalog.Model
        let draftURL: URL
    }

    struct Run {
        let baseline: (
            result: MLXRealModelHarness.GenerationResult,
            events: [MLXGenerationDiagnosticEvent]
        )
        let accelerated: (
            result: MLXRealModelHarness.GenerationResult,
            events: [MLXGenerationDiagnosticEvent]
        )
    }
}
