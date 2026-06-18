extension MLXSession {
    nonisolated func makeIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) throws -> TokenGenerationIterator {
        guard let speculativeDecoding else {
            return .standard(try TokenIterator(
                input: cachePlan.input,
                model: genContext.modelContext.model,
                cache: cachePlan.cache,
                parameters: genContext.parameters,
                processorPrompt: fullInput.text
            ))
        }

        MLXGenerationDiagnostics.recordSpeculativeDecoding(
            numDraftTokens: speculativeDecoding.numDraftTokens
        )
        return .speculative(try SpeculativeTokenIterator(
            input: cachePlan.input,
            mainModel: genContext.modelContext.model,
            draftModel: speculativeDecoding.draftContext.model,
            mainCache: cachePlan.cache,
            draftCache: cachePlan.draftCache,
            parameters: genContext.parameters,
            numDraftTokens: speculativeDecoding.numDraftTokens,
            processorPrompt: fullInput.text
        ))
    }
}
