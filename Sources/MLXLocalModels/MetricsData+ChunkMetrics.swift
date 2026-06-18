extension MetricsData {
    func chunkMetrics(totalDuration: Duration, contextWindowSize: Int?) -> ChunkMetrics {
        let promptProcessingTime = promptStartTime.duration(to: promptEndTime)
        let timeToFirstToken = firstTokenTime.map { generationStartTime.duration(to: $0) }

        return ChunkMetrics(
            timing: TimingMetrics(
                totalTime: totalDuration,
                timeToFirstToken: timeToFirstToken,
                timeSinceLastToken: nil,
                tokenTimings: [],
                promptProcessingTime: promptProcessingTime
            ),
            usage: UsageMetrics(
                generatedTokens: generatedTokenCount,
                totalTokens: promptTokenCount + generatedTokenCount,
                promptTokens: promptTokenCount,
                contextWindowSize: contextWindowSize,
                contextTokensUsed: promptTokenCount + generatedTokenCount,
                kvCacheBytes: kvCacheBytes,
                kvCacheEntries: kvCacheEntries,
                promptCacheReusedTokenCount: promptCacheReusedTokenCount
            ),
            generation: GenerationMetrics(
                stopReason: stopReason,
                temperature: parameters.temperature,
                topP: parameters.topP,
                topK: parameters.topK > 0 ? Int32(parameters.topK) : nil
            )
        )
    }
}
