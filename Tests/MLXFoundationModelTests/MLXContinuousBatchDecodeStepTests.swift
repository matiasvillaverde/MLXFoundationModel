import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch decode step")
struct MLXContinuousBatchDecodeStepTests {
    @Test("exposes stable row identity without touching MLX arrays")
    func exposesStableRowIdentityWithoutArrays() throws {
        let step = try Self.makeDecodeStep(processors: .empty)

        #expect(step.rowCount == 2)
        #expect(step.orderedRowIDs == [100, 200])
    }

    @Test(
        "rejects token batches with the wrong row count",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func rejectsTokenBatchesWithWrongRowCount() throws {
        var step = try Self.makeDecodeStep(processors: .empty)

        do {
            try Device.withDefaultDevice(.cpu) {
                _ = try step.step(previousTokens: MLXArray([1, 2, 3]))
            }
            Issue.record("Expected row-count mismatch")
        } catch MLXContinuousBatchDecodeStepError.rowCountMismatch(let expected, let actual) {
            #expect(expected == 2)
            #expect(actual == 3)
        }
    }

    @Test(
        "rejects scalar cache state before batched decode",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func rejectsScalarCacheStateBeforeBatchedDecode() throws {
        var step = try Self.makeDecodeStep(processors: .empty)
        step.cache = [Self.scalarCache()]

        do {
            try Device.withDefaultDevice(.cpu) {
                _ = try step.step(previousTokens: MLXArray([1, 2]))
            }
            Issue.record("Expected cache row-count mismatch")
        } catch MLXContinuousBatchDecodeStepError.cacheRowCountMismatch(let expected, let actual) {
            #expect(expected == 2)
            #expect(actual == 1)
        }
    }

    @Test(
        "realign filters logit rows and cache state on batch axis",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func realignFiltersLogitRowsAndCacheState() throws {
        var step = try Self.makeDecodeStep(processors: .empty)
        let cache = Self.twoRowCache()
        step.cache = [cache]

        try Device.withDefaultDevice(.cpu) {
            try step.realign(to: [200])
        }

        #expect(step.orderedRowIDs == [200])
        #expect(cache.state.allSatisfy { $0.dim(0) == 1 })
    }

    @Test(
        "runs one batched model forward and samples each row independently",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func runsBatchedForwardAndSamplesRowsIndependently() async throws {
        let recorded = try await Device.withDefaultDevice(.cpu) {
            try await MLXGenerationDiagnostics.withRecording {
                var step = try Self.makeDecodeStep()
                return try step.step(previousTokens: MLXArray([0, 1]))
            }
        }

        #expect(recorded.result.rowIDs == [100, 200])
        #expect(recorded.result.tokenIDs == [2, 1])
        #expect(recorded.result.tokenArray.shape == [2])
        #expect(recorded.events.contains { event in
            guard case .continuousBatchLogits(let snapshot) = event else {
                return false
            }
            return snapshot.rowIDs == [100, 200] && snapshot.tokenIDs == [2, 1]
        })
    }

    private static func makeDecodeStep(
        processors: ProcessorFixture = .masked
    ) throws -> MLXContinuousBatchDecodeStep {
        var rows = MLXContinuousBatchLogitRows()
        try rows.append(
            id: 100,
            row: .init(
                processor: processors.first,
                sampler: ArgMaxSampler()
            )
        )
        try rows.append(
            id: 200,
            row: .init(
                processor: processors.second,
                sampler: ArgMaxSampler()
            )
        )
        return MLXContinuousBatchDecodeStep(
            model: MLXEchoBatchLanguageModel(),
            cache: [],
            logitRows: rows
        )
    }

    private static func scalarCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([1, 1, 1, 1]),
            MLXArray.zeros([1, 1, 1, 1])
        ]
        return cache
    }

    private static func twoRowCache() -> KVCacheSimple {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray.zeros([2, 1, 1, 1]),
            MLXArray.zeros([2, 1, 1, 1])
        ]
        return cache
    }
}
