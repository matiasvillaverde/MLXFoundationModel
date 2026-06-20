import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch generation scheduler")
struct MLXContinuousBatchSchedulerTests {
    @Test("generation row tracks stop and max-token finish policy without arrays")
    func generationRowTracksFinishPolicy() {
        var row = MLXContinuousBatchGenerationRow(
            previousTokenID: 9,
            maximumTokenCount: 2,
            stopTokenIDs: [7]
        )

        #expect(row.accept(tokenID: 3) == nil)
        #expect(row.previousTokenID == 3)
        #expect(row.generatedTokenCount == 1)
        #expect(row.accept(tokenID: 7) == .stopToken(7))
        #expect(row.previousTokenID == 7)
        #expect(row.generatedTokenCount == 2)
    }

    @Test("rejects decoder row drift before allocating MLX token arrays")
    func rejectsDecoderRowDriftWithoutArrays() throws {
        var coordinator = MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>()
        _ = try coordinator.admit(.init(previousTokenID: 1, maximumTokenCount: 4))
        var scheduler = MLXContinuousBatchGenerationScheduler(
            coordinator: coordinator,
            decoder: ScriptedContinuousBatchDecoder(rowIDs: [99], tokenBatches: [])
        )

        do {
            _ = try scheduler.step()
            Issue.record("Expected row mismatch")
        } catch MLXContinuousBatchSchedulerError.decoderRowMismatch(
            let expected,
            let actual
        ) {
            #expect(expected == [0])
            #expect(actual == [99])
        }
    }

    @Test("stream-requested finish removes rows and realigns decoder")
    func streamRequestedFinishRemovesRowsAndRealignsDecoder() throws {
        var coordinator = MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>()
        let first = try coordinator.admit(.init(previousTokenID: 11, maximumTokenCount: 4))
        let second = try coordinator.admit(.init(
            previousTokenID: 20,
            maximumTokenCount: 4,
            generatedTokenCount: 2
        ))
        var scheduler = MLXContinuousBatchGenerationScheduler(
            coordinator: coordinator,
            decoder: ScriptedContinuousBatchDecoder(
                rowIDs: [first, second],
                tokenBatches: []
            )
        )

        let finishedRows = try scheduler.finishRows(
            [second],
            reason: .streamRequestedStop(20)
        )

        #expect(finishedRows == [
            .init(
                rowID: second,
                tokenID: 20,
                generatedTokenCount: 2,
                reason: .streamRequestedStop(20)
            )
        ])
        #expect(scheduler.orderedRowIDs == [first])
        #expect(scheduler.decoder.orderedRowIDs == [first])
    }

    @Test(
        "accepts sampled tokens and removes finished rows",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func acceptsSampledTokensAndRemovesFinishedRows() throws {
        var coordinator = MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>()
        let first = try coordinator.admit(.init(
            previousTokenID: 11,
            maximumTokenCount: 3,
            stopTokenIDs: [2]
        ))
        let second = try coordinator.admit(.init(
            previousTokenID: 12,
            maximumTokenCount: 1
        ))
        var scheduler = MLXContinuousBatchGenerationScheduler(
            coordinator: coordinator,
            decoder: ScriptedContinuousBatchDecoder(
                rowIDs: [first, second],
                tokenBatches: [[2, 4]]
            )
        )

        let result = try Device.withDefaultDevice(.cpu) {
            try scheduler.step()
        }

        #expect(result.sampledTokens.rowIDs == [first, second])
        #expect(result.sampledTokens.tokenIDs == [2, 4])
        #expect(result.finishedRows == [
            .init(rowID: first, tokenID: 2, generatedTokenCount: 1, reason: .stopToken(2)),
            .init(rowID: second, tokenID: 4, generatedTokenCount: 1, reason: .maximumTokenCount)
        ])
        #expect(result.activeRowIDs.isEmpty)
        #expect(scheduler.orderedRowIDs.isEmpty)
        #expect(scheduler.decoder.orderedRowIDs.isEmpty)
    }
}
