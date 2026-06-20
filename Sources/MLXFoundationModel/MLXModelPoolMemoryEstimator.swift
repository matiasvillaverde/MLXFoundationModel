enum MLXModelPoolMemoryEstimator {
    static func estimatedResidentBytes(for model: MLXLanguageModel) -> Int {
        MLXModelWeightArtifactScanner.residentWeightBytes(in: model.model.location)
    }
}
