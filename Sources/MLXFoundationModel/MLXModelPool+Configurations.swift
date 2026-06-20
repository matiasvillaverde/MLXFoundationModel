import MLXLocalModels

extension MLXModelPool {
    func residentConfiguration(for model: MLXLanguageModel) -> ProviderConfiguration {
        let configuration = model.providerConfiguration
        let modelName = baseModelID(for: model.model.id)
        guard modelName != configuration.modelName else {
            return configuration
        }
        return ProviderConfiguration(
            location: configuration.location,
            authentication: configuration.authentication,
            modelName: modelName,
            compute: configuration.compute,
            runtime: configuration.runtime
        )
    }

    func baseModelID(for publicID: String) -> String {
        servingProfileTargets[publicID] ?? publicID
    }
}
