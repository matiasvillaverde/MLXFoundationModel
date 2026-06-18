extension MLXSession {
    internal func createGenerateParameters(
        from sampling: SamplingParameters,
        limits: ResourceLimits
    ) -> GenerateParameters {
        let penaltyContextSize = sampling.repetitionPenaltyRange ?? 64

        return GenerateParameters(
            maxTokens: limits.maxTokens,
            maxKVSize: limits.maxKVSize,
            kvBits: limits.kvCacheBits,
            kvGroupSize: limits.kvCacheGroupSize,
            quantizedKVStart: limits.quantizedKVStart,
            temperature: sampling.temperature,
            topP: sampling.topP,
            topK: sampling.topK ?? 0,
            minP: sampling.advanced.minP ?? 0.0,
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
            prefillStepSize: limits.prefillStepSize
        )
    }
}
