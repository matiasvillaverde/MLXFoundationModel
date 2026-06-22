extension MetricsData {
    func requestSummary(
        totalDuration: Duration,
        modelName: String?,
        strategy: MLXGenerationExecutionStrategy?
    ) -> MLXRequestSummary {
        let totalDurationSeconds = totalDuration.mlxSeconds
        let promptProcessingSeconds = promptStartTime.duration(to: promptEndTime).mlxSeconds
        let timeToFirstTokenSeconds = firstTokenTime.map { firstTokenInstant in
            generationStartTime.duration(to: firstTokenInstant).mlxSeconds
        }
        return requestSummary(
            timing: RequestSummaryTiming(
                generationDurationSeconds: totalDurationSeconds - promptProcessingSeconds,
                promptProcessingSeconds: promptProcessingSeconds,
                timeToFirstTokenSeconds: timeToFirstTokenSeconds,
                totalDurationSeconds: totalDurationSeconds
            ),
            modelName: modelName,
            strategy: strategy
        )
    }

    private func requestSummary(
        timing: RequestSummaryTiming,
        modelName: String?,
        strategy: MLXGenerationExecutionStrategy?
    ) -> MLXRequestSummary {
        let totalTokens = promptTokenCount + generatedTokenCount

        return MLXRequestSummary(
            modelName: modelName,
            strategy: strategy.map(String.init(describing:)),
            promptTokens: promptTokenCount,
            generatedTokens: generatedTokenCount,
            totalTokens: totalTokens,
            cachedPromptTokens: promptCacheReusedTokenCount,
            totalDurationSeconds: timing.totalDurationSeconds,
            timeToFirstTokenSeconds: timing.timeToFirstTokenSeconds,
            promptProcessingSeconds: timing.promptProcessingSeconds,
            promptTokensPerSecond: rate(Double(promptTokenCount), over: timing.promptProcessingSeconds),
            generationTokensPerSecond: rate(
                Double(generatedTokenCount),
                over: max(timing.generationDurationSeconds, 0)
            ),
            totalTokensPerSecond: rate(Double(totalTokens), over: timing.totalDurationSeconds),
            kvCacheBytes: kvCacheBytes,
            kvCacheEntries: kvCacheEntries,
            stopReason: stopReason.rawValue,
            temperature: Double(parameters.temperature),
            topP: Double(parameters.topP),
            topK: parameters.topK > 0 ? parameters.topK : nil,
            grammarKind: parameters.grammar?.kind.rawValue
        )
    }

    private func rate(_ count: Double, over seconds: Double) -> Double? {
        guard seconds > 0 else {
            return nil
        }
        return count / seconds
    }
}

private struct RequestSummaryTiming {
    let generationDurationSeconds: Double
    let promptProcessingSeconds: Double
    let timeToFirstTokenSeconds: Double?
    let totalDurationSeconds: Double
}
