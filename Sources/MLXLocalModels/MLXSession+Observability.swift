import Foundation

extension MLXSession {
    func recordGenerationSummary(
        metricsData: MetricsData,
        totalDuration: Duration
    ) {
        let summary = metricsData.requestSummary(
            totalDuration: totalDuration,
            modelName: configuration?.modelName,
            strategy: generationExecutionPlan?.selectedStrategy
        )
        MLXObservability.recordRequestSummary(summary)
    }
}
