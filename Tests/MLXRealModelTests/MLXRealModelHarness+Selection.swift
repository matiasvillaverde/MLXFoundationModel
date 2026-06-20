@testable import MLXLocalModels
import Testing

extension MLXRealModelHarness {
    static func selectedModel(
        _ id: String,
        in models: [MLXRealModelCatalog.Model]
    ) throws -> MLXRealModelCatalog.Model? {
        let model = try #require(models.first { $0.id == id })
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
        guard selected.contains(where: { $0.id == id }) else {
            return nil
        }
        #expect(
            MLXRealModelEnvironment.hasModelFiles(for: model),
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage([model]))
        )
        return model
    }
}
