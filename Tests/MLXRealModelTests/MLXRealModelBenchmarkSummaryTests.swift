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
}
