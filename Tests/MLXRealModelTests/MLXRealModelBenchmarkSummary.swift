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

    func benchmarkJSONLine(model: MLXRealModelCatalog.Model) -> String {
        "BENCH_JSON \(encodedRecord(model: model, kind: "bench", iteration: nil))"
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

    func stressJSONLine(model: MLXRealModelCatalog.Model, iteration: Int) -> String {
        "STRESS_JSON \(encodedRecord(model: model, kind: "stress", iteration: iteration))"
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

    private func encodedRecord(
        model: MLXRealModelCatalog.Model,
        kind: String,
        iteration: Int?
    ) -> String {
        let record = BenchmarkRecord(
            schemaVersion: 1,
            kind: kind,
            model: model.id,
            architecture: model.architecture,
            iteration: iteration,
            generatedTokens: generatedTokens,
            promptTokens: promptTokens,
            totalTokens: totalTokens,
            totalSeconds: totalSeconds,
            promptSeconds: promptSeconds,
            decodeSeconds: decodeSeconds,
            promptTokensPerSecond: promptTokensPerSecond,
            decodeTokensPerSecond: decodeTokensPerSecond,
            endToEndGeneratedTokensPerSecond: endToEndGeneratedTokensPerSecond,
            totalTokensPerSecond: totalTokensPerSecond
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(record),
              let json = String(data: data, encoding: .utf8) else {
            return "{}"
        }
        return json
    }

    private static func seconds(_ duration: Duration) -> Double {
        let components = duration.components
        return Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}

private struct BenchmarkRecord: Encodable {
    let schemaVersion: Int
    let kind: String
    let model: String
    let architecture: String
    let iteration: Int?
    let generatedTokens: Int
    let promptTokens: Int
    let totalTokens: Int
    let totalSeconds: Double
    let promptSeconds: Double
    let decodeSeconds: Double
    let promptTokensPerSecond: Double
    let decodeTokensPerSecond: Double
    let endToEndGeneratedTokensPerSecond: Double
    let totalTokensPerSecond: Double

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case kind
        case model
        case architecture
        case iteration
        case generatedTokens = "generated_tokens"
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
        case totalSeconds = "total_seconds"
        case promptSeconds = "prompt_seconds"
        case decodeSeconds = "decode_seconds"
        case promptTokensPerSecond = "prompt_tps"
        case decodeTokensPerSecond = "decode_tps"
        case endToEndGeneratedTokensPerSecond = "e2e_tps"
        case totalTokensPerSecond = "total_tps"
    }
}
