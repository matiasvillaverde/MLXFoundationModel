import Foundation
import MLXLocalModels
import Testing

@Suite("MLX real-model benchmark summary")
struct MLXRealModelBenchmarkSummaryTests {
    @Test("benchmark line includes prompt decode e2e and total throughput")
    func benchmarkLineIncludesThroughputBreakdown() throws {
        let summary = try #require(MLXRealModelBenchmarkSummary(metrics: Self.metrics))
        let line = summary.benchmarkLine(model: Self.model)

        #expect(line.contains("BENCH model=fixture-model architecture=fixture"))
        #expect(line.contains("generated=4 prompt=10 total=14"))
        #expect(line.contains("total_s=2.0000 prompt_s=1.0000 decode_s=1.0000"))
        #expect(line.contains("prompt_tps=10.00"))
        #expect(line.contains("decode_tps=4.00"))
        #expect(line.contains("e2e_tps=2.00"))
        #expect(line.contains("total_tps=7.00"))
    }

    @Test("benchmark JSON line exposes stable machine-readable throughput")
    func benchmarkJSONLineExposesStableThroughput() throws {
        let summary = try #require(MLXRealModelBenchmarkSummary(metrics: Self.metrics))
        let json = try Self.jsonPayload(from: summary.benchmarkJSONLine(model: Self.model), prefix: "BENCH_JSON ")

        #expect(json["schema_version"] as? Int == 1)
        #expect(json["kind"] as? String == "bench")
        #expect(json["model"] as? String == "fixture-model")
        #expect(json["architecture"] as? String == "fixture")
        #expect(json["generated_tokens"] as? Int == 4)
        #expect(json["prompt_tokens"] as? Int == 10)
        #expect(json["total_tokens"] as? Int == 14)
        #expect(json["prompt_tps"] as? Double == 10)
        #expect(json["decode_tps"] as? Double == 4)
        #expect(json["e2e_tps"] as? Double == 2)
        #expect(json["total_tps"] as? Double == 7)
        #expect(json["iteration"] == nil)
    }

    @Test("stress line includes the iteration and throughput breakdown")
    func stressLineIncludesIterationAndThroughputBreakdown() throws {
        let summary = try #require(MLXRealModelBenchmarkSummary(metrics: Self.metrics))
        let line = summary.stressLine(model: Self.model, iteration: 3)

        #expect(line.contains("STRESS model=fixture-model architecture=fixture iteration=3"))
        #expect(line.contains("prompt_tps=10.00"))
        #expect(line.contains("decode_tps=4.00"))
        #expect(line.contains("e2e_tps=2.00"))
        #expect(line.contains("total_tps=7.00"))
    }

    @Test("stress JSON line includes iteration")
    func stressJSONLineIncludesIteration() throws {
        let summary = try #require(MLXRealModelBenchmarkSummary(metrics: Self.metrics))
        let json = try Self.jsonPayload(
            from: summary.stressJSONLine(model: Self.model, iteration: 3),
            prefix: "STRESS_JSON "
        )

        #expect(json["kind"] as? String == "stress")
        #expect(json["iteration"] as? Int == 3)
        #expect(json["decode_tps"] as? Double == 4)
    }

    private static let metrics = ChunkMetrics(
        timing: TimingMetrics(
            totalTime: .seconds(2),
            promptProcessingTime: .seconds(1)
        ),
        usage: UsageMetrics(
            generatedTokens: 4,
            totalTokens: 14,
            promptTokens: 10
        )
    )

    private static let model = MLXRealModelCatalog.Model(
        id: "fixture-model",
        displayName: "Fixture",
        architecture: "fixture",
        repository: nil,
        relativePath: "fixture",
        prompt: "Prompt",
        expectedTokens: [],
        maxTokens: 4,
        minimumMemoryGB: nil,
        minimumDiskGB: nil,
        memoryGuardTier: nil,
        tags: []
    )

    private static func jsonPayload(
        from line: String,
        prefix: String
    ) throws -> [String: Any] {
        #expect(line.hasPrefix(prefix))
        let payload = String(line.dropFirst(prefix.count))
        let data = try #require(payload.data(using: .utf8))
        let object = try JSONSerialization.jsonObject(with: data)
        return try #require(object as? [String: Any])
    }
}
