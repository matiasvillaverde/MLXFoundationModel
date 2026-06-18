extension MLXSession {
    nonisolated func makeIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) throws -> TokenGenerationIterator {
        if genContext.parameters.grammar != nil,
            genContext.modelContext.grammarCompiler == nil,
            let error = genContext.modelContext.grammarCompilerError {
            recordGrammarIteratorEvent(
                stage: .compilerUnavailable,
                grammar: genContext.parameters.grammar,
                message: error.localizedDescription
            )
            throw error
        }

        guard let speculativeDecoding, genContext.parameters.grammar == nil else {
            recordSpeculativeBypassIfNeeded(
                speculativeDecoding: speculativeDecoding,
                grammar: genContext.parameters.grammar
            )
            return .standard(try makeStandardIterator(
                genContext: genContext,
                cachePlan: cachePlan,
                fullInput: fullInput
            ))
        }

        MLXGenerationDiagnostics.recordSpeculativeDecoding(
            numDraftTokens: speculativeDecoding.numDraftTokens
        )
        return .speculative(try makeSpeculativeIterator(
            genContext: genContext,
            cachePlan: cachePlan,
            fullInput: fullInput,
            speculativeDecoding: speculativeDecoding
        ))
    }

    nonisolated private func makeStandardIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput
    ) throws -> TokenIterator {
        try TokenIterator(
            input: cachePlan.input,
            model: genContext.modelContext.model,
            cache: cachePlan.cache,
            parameters: genContext.parameters,
            processorPrompt: fullInput.text,
            grammarCompiler: genContext.modelContext.grammarCompiler
        )
    }

    nonisolated private func makeSpeculativeIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration
    ) throws -> SpeculativeTokenIterator {
        try SpeculativeTokenIterator(
            input: cachePlan.input,
            mainModel: genContext.modelContext.model,
            draftModel: speculativeDecoding.draftContext.model,
            mainCache: cachePlan.cache,
            draftCache: cachePlan.draftCache,
            parameters: genContext.parameters,
            numDraftTokens: speculativeDecoding.numDraftTokens,
            processorPrompt: fullInput.text,
            grammarCompiler: genContext.modelContext.grammarCompiler
        )
    }

    nonisolated private func recordSpeculativeBypassIfNeeded(
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        grammar: GrammarSamplingConfiguration?
    ) {
        guard speculativeDecoding != nil, let grammar else {
            return
        }
        recordGrammarIteratorEvent(
            stage: .speculativeBypassed,
            grammar: grammar,
            message: "Speculative decoding is disabled while token-level grammar constraints are active"
        )
    }

    nonisolated private func recordGrammarIteratorEvent(
        stage: MLXGrammarConstraintSnapshot.Stage,
        grammar: GrammarSamplingConfiguration?,
        message: String
    ) {
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: stage,
            kind: grammar?.kind,
            mode: nil,
            tokenCount: nil,
            tokenID: nil,
            vocabularySize: nil,
            bitmaskSize: nil,
            isCompleted: nil,
            isTerminated: nil,
            message: message
        ))
    }
}
