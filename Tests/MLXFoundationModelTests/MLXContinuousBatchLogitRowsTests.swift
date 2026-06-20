import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch logit rows")
struct MLXContinuousBatchLogitRowsTests {
    @Test("keeps logit rows aligned after removal")
    func keepsLogitRowsAlignedAfterRemoval() throws {
        var rows = MLXContinuousBatchLogitRows()
        try rows.append(id: 1, row: .init(processor: nil, sampler: ArgMaxSampler()))
        try rows.append(id: 2, row: .init(processor: nil, sampler: ArgMaxSampler()))

        _ = rows.remove(id: 1)

        #expect(rows.orderedRowIDs == [2])
        #expect(rows[2]?.generatedTokenCount == 0)
    }

    @Test(
        "rejects logits with the wrong row count",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func rejectsLogitsWithWrongRowCount() throws {
        var rows = MLXContinuousBatchLogitRows()
        try rows.append(id: 7, row: .init(processor: nil, sampler: ArgMaxSampler()))

        do {
            try Device.withDefaultDevice(.cpu) {
                _ = try rows.sample(logits: MLXArray.zeros([2, 3]))
            }
            Issue.record("Expected row-count mismatch")
        } catch MLXContinuousBatchLogitRowsError.rowCountMismatch(let expected, let actual) {
            #expect(expected == 1)
            #expect(actual == 2)
        }
    }

    @Test(
        "samples each row with its own processor",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func samplesEachRowWithItsOwnProcessor() async throws {
        let recorded = try await Device.withDefaultDevice(.cpu) {
            try await MLXGenerationDiagnostics.withRecording {
                var rows = MLXContinuousBatchLogitRows()
                try Self.appendMaskedRows(to: &rows)
                let logits = MLXArray([
                    Float(0), Float(10), Float(3),
                    Float(9), Float(8), Float(1)
                ]).reshaped([2, 3])

                return try rows.sample(logits: logits)
            }
        }

        #expect(recorded.result.rowIDs == [10, 20])
        #expect(recorded.result.tokenIDs == [2, 1])
        let snapshots: [MLXContinuousBatchLogitsSnapshot] = recorded.events.compactMap { event in
            guard case .continuousBatchLogits(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
        #expect(snapshots.last?.rowIDs == [10, 20])
        #expect(snapshots.last?.tokenIDs == [2, 1])
    }

    private static func appendMaskedRows(
        to rows: inout MLXContinuousBatchLogitRows
    ) throws {
        try rows.append(
            id: 10,
            row: .init(
                processor: SuppressTokensProcessor(tokenIds: [1]),
                sampler: ArgMaxSampler()
            )
        )
        try rows.append(
            id: 20,
            row: .init(
                processor: SuppressTokensProcessor(tokenIds: [0]),
                sampler: ArgMaxSampler()
            )
        )
    }
}
