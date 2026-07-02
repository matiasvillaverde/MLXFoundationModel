import Foundation
import MLXLocalModels

struct MLXRealModelBenchmarkSummary {
    let generatedTokens: Int
    let promptTokens: Int
    let totalTokens: Int
    let totalSeconds: Double
    let promptSeconds: Double
    let decodeSeconds: Double

    init?(metrics: ChunkMetrics) {
        guard let usage = metrics.usage,
              let timing = metrics.timing,
              usage.generatedTokens > 0 else {
            return nil
        }

        generatedTokens = usage.generatedTokens
        promptTokens = usage.promptTokens ?? 0
        totalTokens = usage.totalTokens
        totalSeconds = Self.seconds(timing.totalTime)
        promptSeconds = timing.promptProcessingTime.map(Self.seconds) ?? 0
        decodeSeconds = max(totalSeconds - promptSeconds, 0)
    }

    func benchmarkLine(model: MLXRealModelCatalog.Model) -> String {
        String(
            format: """
            BENCH model=%@ architecture=%@ generated=%d prompt=%d total=%d \
            total_s=%.4f prompt_s=%.4f decode_s=%.4f \
            prompt_tps=%.2f decode_tps=%.2f e2e_tps=%.2f total_tps=%.2f
            """,
            model.id,
            model.architecture,
            generatedTokens,
            promptTokens,
            totalTokens,
            totalSeconds,
            promptSeconds,
            decodeSeconds,
            promptTokensPerSecond,
            decodeTokensPerSecond,
            endToEndGeneratedTokensPerSecond,
            totalTokensPerSecond
        )
    }

    func stressLine(model: MLXRealModelCatalog.Model, iteration: Int) -> String {
        String(
            format: """
            STRESS model=%@ architecture=%@ iteration=%d generated=%d prompt=%d total=%d \
            total_s=%.4f prompt_s=%.4f decode_s=%.4f \
            prompt_tps=%.2f decode_tps=%.2f e2e_tps=%.2f total_tps=%.2f
            """,
            model.id,
            model.architecture,
            iteration,
            generatedTokens,
            promptTokens,
            totalTokens,
            totalSeconds,
            promptSeconds,
            decodeSeconds,
            promptTokensPerSecond,
            decodeTokensPerSecond,
            endToEndGeneratedTokensPerSecond,
            totalTokensPerSecond
        )
    }

    private var promptTokensPerSecond: Double {
        rate(Double(promptTokens), over: promptSeconds)
    }

    private var decodeTokensPerSecond: Double {
        rate(Double(generatedTokens), over: decodeSeconds)
    }

    private var endToEndGeneratedTokensPerSecond: Double {
        rate(Double(generatedTokens), over: totalSeconds)
    }

    private var totalTokensPerSecond: Double {
        rate(Double(totalTokens), over: totalSeconds)
    }

    private func rate(_ count: Double, over seconds: Double) -> Double {
        guard seconds > 0 else {
            return 0
        }
        return count / seconds
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
