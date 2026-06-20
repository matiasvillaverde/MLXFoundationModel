@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch assembler")
struct MLXContinuousBatchAssemblerTests {
    @Test("assembles aligned coordinator rows and stream rows")
    func assemblesAlignedCoordinatorRowsAndStreamRows() throws {
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()

        let batch = try MLXContinuousBatchAssembler.assemble(requests: [
            Self.request(previousTokenID: 10, maximumTokenCount: 2, sink: firstSink),
            Self.request(previousTokenID: 20, maximumTokenCount: 3, sink: secondSink)
        ])

        #expect(batch.count == 2)
        #expect(batch.orderedRowIDs == [0, 1])
        #expect(batch.streamRows.map(\.id) == batch.orderedRowIDs)
        #expect(batch.coordinator[0]?.previousTokenID == 10)
        #expect(batch.coordinator[1]?.maximumTokenCount == 3)

        _ = batch.streamRows[0].stream(tokenID: 7)
        _ = batch.streamRows[1].stream(tokenID: 8)

        #expect(firstSink.texts() == ["token-7"])
        #expect(secondSink.texts() == ["token-8"])
        #expect(firstSink.tokenIDs() == [[7]])
        #expect(secondSink.tokenIDs() == [[8]])
    }

    @Test("preserves custom stream token handlers")
    func preservesCustomStreamTokenHandlers() throws {
        let sink = RecordingBatchSink()
        let batch = try MLXContinuousBatchAssembler.assemble(requests: [
            Self.stoppingRequest(previousTokenID: 10, sink: sink)
        ])

        let disposition = batch.streamRows[0].stream(tokenID: 42)

        #expect(sink.texts() == ["custom-42"])
        guard case .finish(.streamRequestedStop(42)) = disposition else {
            Issue.record("Expected stream-requested stop disposition")
            return
        }
    }

    @Test("rejects empty request batches")
    func rejectsEmptyRequestBatches() throws {
        #expect(throws: MLXContinuousBatchCoordinatorError.self) {
            _ = try MLXContinuousBatchAssembler.assemble(requests: [])
        }
    }

    private static func request(
        previousTokenID: Int,
        maximumTokenCount: Int,
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchGenerationRequest {
        MLXContinuousBatchGenerationRequest(
            generationRow: .init(
                previousTokenID: previousTokenID,
                maximumTokenCount: maximumTokenCount
            ),
            tokenText: { "token-\($0)" },
            sink: sink.streamSink()
        )
    }

    private static func stoppingRequest(
        previousTokenID: Int,
        sink: RecordingBatchSink
    ) -> MLXContinuousBatchGenerationRequest {
        MLXContinuousBatchGenerationRequest(
            generationRow: .init(
                previousTokenID: previousTokenID,
                maximumTokenCount: 4
            ),
            sink: sink.streamSink()
        ) { tokenID, sink in
            sink.yield(LLMStreamChunk(
                text: "custom-\(tokenID)",
                event: .text,
                tokenCount: 1,
                tokenIDs: [tokenID]
            ))
            return .finish(.streamRequestedStop(tokenID))
        }
    }
}
