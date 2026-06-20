extension MLXSession {
    nonisolated func makeIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput,
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?
    ) throws -> TokenGenerationIterator {
        try throwGrammarCompilerErrorIfNeeded(genContext)

        if shouldUseNativeMTP(genContext) {
            return try makeNativeMTPIterator(
                genContext: genContext,
                cachePlan: cachePlan,
                fullInput: fullInput
            )
        }

        guard let speculativeDecoding, genContext.parameters.grammar == nil else {
            recordSpeculativeBypassIfNeeded(
                speculativeDecoding: speculativeDecoding,
                nativeMTPRequested: isNativeMTPRequested(genContext),
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

    nonisolated private func throwGrammarCompilerErrorIfNeeded(
        _ genContext: GenerationContext
    ) throws {
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
    }

    nonisolated private func shouldUseNativeMTP(_ genContext: GenerationContext) -> Bool {
        isNativeMTPRequested(genContext) && genContext.parameters.grammar == nil
    }

    nonisolated private func isNativeMTPRequested(_ genContext: GenerationContext) -> Bool {
        genContext.runtimePreferences.optimization.mode == .nativeMTP
    }

    nonisolated private func makeNativeMTPIterator(
        genContext: GenerationContext,
        cachePlan: PromptCachePlan,
        fullInput: LMInput
    ) throws -> TokenGenerationIterator {
        guard
            let model = genContext.modelContext.model as? any NativeMTPModel,
            model.supportsNativeMTP
        else {
            throw LLMError.invalidConfiguration(
                "nativeMTP was requested, but the loaded model does not expose native MTP heads."
            )
        }

        let numDraftTokens = max(1, genContext.runtimePreferences.speculativeDraftTokens)
        MLXGenerationDiagnostics.recordSpeculativeDecoding(numDraftTokens: numDraftTokens)
        return .nativeMTP(try NativeMTPTokenIterator(
            input: cachePlan.input,
            model: model,
            cache: cachePlan.cache,
            parameters: genContext.parameters,
            numDraftTokens: numDraftTokens,
            processorPrompt: fullInput.text,
            grammarCompiler: genContext.modelContext.grammarCompiler
        ))
    }

    nonisolated private func recordSpeculativeBypassIfNeeded(
        speculativeDecoding: MLXSpeculativeDecodingConfiguration?,
        nativeMTPRequested: Bool,
        grammar: GrammarSamplingConfiguration?
    ) {
        guard speculativeDecoding != nil || nativeMTPRequested, let grammar else {
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
