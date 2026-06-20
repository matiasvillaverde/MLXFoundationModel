import MLXLocalModels

extension MLXModelPool {
    func residentKeys(
        matchingPublicID publicID: String,
        modelID: String
    ) -> [ProviderConfiguration] {
        residents.keys.filter { key in
            guard let entry = residents[key] else {
                return false
            }
            return model(entry.model, matchesPublicID: publicID, modelID: modelID)
        }
    }

    func loadingResidentKeys(
        matchingPublicID publicID: String,
        modelID: String
    ) -> [ProviderConfiguration] {
        loadingResidents.keys.filter { key in
            guard let entry = loadingResidents[key] else {
                return false
            }
            return model(entry.model, matchesPublicID: publicID, modelID: modelID)
        }
    }

    func model(
        _ model: MLXLanguageModel,
        matchesPublicID publicID: String,
        modelID: String
    ) -> Bool {
        model.model.id == publicID ||
            (publicID == modelID && baseModelID(for: model.model.id) == modelID)
    }
}
