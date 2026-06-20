import MLX

internal struct MLXContinuousBatchModelPrefillRunner: MLXContinuousBatchPrefillRunning {
    internal let grammarCompiler: GrammarConstraintCompiler?
    internal let model: any LanguageModel

    internal init(
        model: any LanguageModel,
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) {
        self.grammarCompiler = grammarCompiler
        self.model = model
    }

    internal func run(
        requests: [MLXContinuousBatchPrefillRequest]
    ) throws -> MLXContinuousBatchPrefillResult {
        try MLXContinuousBatchPrefill.run(
            model: model,
            requests: requests,
            grammarCompiler: grammarCompiler
        )
    }
}
