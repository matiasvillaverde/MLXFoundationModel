extension MLXSession {
    internal func createGenerateParameters(
        from input: LLMInput,
        container: ModelContainer,
        runtimePreferences: ModelRuntimePreferences
    ) async -> GenerateParameters {
        let tokenSamplingConfiguration = await makeTokenSamplingConfiguration(
            from: container,
            drySequenceBreakers: input.sampling.advanced.dry?.sequenceBreakers ?? [],
            reasoningEndMarker: input.sampling.advanced.reasoningBudget?.endMarker
        )
        return createGenerateParameters(
            from: input.sampling,
            limits: input.limits,
            runtimePreferences: runtimePreferences,
            suppressTokenIds: tokenSamplingConfiguration.suppressTokenIds,
            xtcProtectedTokenIds: tokenSamplingConfiguration.xtcProtectedTokenIds,
            drySequenceBreakerTokenIds: tokenSamplingConfiguration.drySequenceBreakerTokenIds,
            reasoningEndTokenIds: tokenSamplingConfiguration.reasoningEndTokenIds
        )
    }

    internal func createGenerateParameters(
        from sampling: SamplingParameters,
        limits: ResourceLimits,
        runtimePreferences: ModelRuntimePreferences = .default,
        suppressTokenIds: Set<Int> = [],
        xtcProtectedTokenIds: Set<Int> = [],
        drySequenceBreakerTokenIds: [[Int]] = [],
        reasoningEndTokenIds: [Int] = []
    ) -> GenerateParameters {
        let penaltyContextSize = sampling.repetitionPenaltyRange ?? 64
        let xtc = sampling.advanced.xtc
        let kvCacheBits = resolvedKVCacheBits(
            limits: limits,
            runtimePreferences: runtimePreferences
        )

        return GenerateParameters(
            maxTokens: limits.maxTokens,
            maxKVSize: limits.maxKVSize,
            kvBits: kvCacheBits,
            kvGroupSize: limits.kvCacheGroupSize,
            quantizedKVStart: limits.quantizedKVStart,
            quantizedKVSkipLastLayer: resolvedQuantizedKVSkipLastLayer(
                limits: limits,
                runtimePreferences: runtimePreferences
            ),
            indexCacheFrequency: runtimePreferences.optimization.indexCacheFrequency,
            temperature: sampling.temperature,
            topP: sampling.topP,
            topK: sampling.topK ?? 0,
            minP: sampling.advanced.minP ?? 0.0,
            typicalP: sampling.advanced.typicalP,
            topNSigma: sampling.advanced.topNSigma,
            xtcProbability: xtc?.probability ?? 0.0,
            xtcThreshold: xtc?.threshold ?? 0.1,
            xtcMinKeep: xtc?.minKeep ?? 1,
            xtcProtectedTokenIds: xtcProtectedTokenIds,
            mirostat: sampling.advanced.mirostat,
            dry: sampling.advanced.dry,
            drySequenceBreakerTokenIds: drySequenceBreakerTokenIds,
            adaptiveP: sampling.advanced.adaptiveP,
            repetitionPenalty: sampling.repetitionPenalty,
            repetitionContextSize: penaltyContextSize,
            presencePenalty: sampling.presencePenalty,
            presenceContextSize: penaltyContextSize,
            frequencyPenalty: sampling.frequencyPenalty,
            frequencyContextSize: penaltyContextSize,
            seed: sampling.seed,
            grammar: sampling.advanced.grammar,
            logitBias: sampling.advanced.logitBias.reduce(into: [:]) { result, item in
                result[Int(item.key)] = item.value
            },
            reasoningBudgetTokens: sampling.advanced.reasoningBudget?.maximumTokens,
            reasoningEndTokenIds: reasoningEndTokenIds,
            suppressTokenIds: suppressTokenIds,
            prefillStepSize: limits.prefillStepSize,
            promptCacheReuseAlignment: runtimePreferences.promptCacheReuseAlignment
        )
    }

    private func resolvedKVCacheBits(
        limits: ResourceLimits,
        runtimePreferences: ModelRuntimePreferences
    ) -> Int? {
        limits.kvCacheBits ?? runtimeKVCacheBits(runtimePreferences.optimization)
    }

    private func resolvedQuantizedKVSkipLastLayer(
        limits: ResourceLimits,
        runtimePreferences: ModelRuntimePreferences
    ) -> Bool {
        guard limits.kvCacheBits == nil else {
            return false
        }
        let optimization = runtimePreferences.optimization
        return optimization.mode == .turboQuantKV && optimization.turboQuantSkipLastLayer
    }

    private func runtimeKVCacheBits(_ optimization: MLXRuntimeOptimizationConfiguration) -> Int? {
        guard optimization.mode == .turboQuantKV else {
            return nil
        }
        return Int(optimization.turboQuantKVBits.rounded(.up))
    }
}
