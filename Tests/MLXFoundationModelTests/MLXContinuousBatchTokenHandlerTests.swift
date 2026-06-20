@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch token handler")
struct MLXContinuousBatchTokenHandlerTests {
    @Test("streams through shared stop-sequence emitter")
    func streamsThroughSharedStopSequenceEmitter() {
        let sink = RecordingBatchSink()
        let state = GenerationState()
        state.stopDetector = StopSequenceDetector(sequences: ["STOP"])
        let handler = Self.handler(state: state, sink: sink) { tokenID in
            tokenID == 1 ? "hello ST" : "OP hidden"
        }

        #expect(handler.stream(tokenID: 1) == .streamed)
        #expect(handler.stream(tokenID: 2) == .finish(.streamRequestedStop(2)))

        #expect(sink.texts() == ["hello", " "])
        #expect(sink.tokenIDs() == [[1], [2]])
        #expect(state.allTokens == [1, 2])
        #expect(state.generatedText == "hello ")
        #expect(state.generatedTokenCount == 2)
        #expect(state.stopReason == .stopSequence)
    }

    @Test("suppresses pending stop-prefix text until flush")
    func suppressesPendingStopPrefixUntilFlush() {
        let sink = RecordingBatchSink()
        let state = GenerationState()
        state.stopDetector = StopSequenceDetector(sequences: ["STOP"])
        let handler = Self.handler(state: state, sink: sink) { _ in
            "hel"
        }

        #expect(handler.stream(tokenID: 1) == .suppressed)
        #expect(sink.texts().isEmpty)
        handler.flush()

        #expect(sink.texts() == ["hel"])
        #expect(sink.tokenIDs() == [[1]])
        #expect(state.generatedText == "hel")
    }

    @Test("prefill request flushes pending text when row finishes")
    func prefillRequestFlushesPendingTextWhenRowFinishes() {
        let sink = RecordingBatchSink()
        let state = GenerationState()
        state.stopDetector = StopSequenceDetector(sequences: ["STOP"])
        let request = MLXContinuousBatchPrefillRequest(
            promptTokenIDs: [10],
            parameters: GenerateParameters(maxTokens: 1),
            state: state,
            decodeToken: { _ in "hel" },
            sink: sink.streamSink()
        )

        #expect(request.stream(tokenID: 1) == .suppressed)
        let generationRequest = request.continuousBatchGenerationRequest(
            previousTokenID: 1,
            generatedTokenCount: 1
        )
        generationRequest.sink.finish(.maximumTokenCount)

        #expect(sink.texts() == ["hel"])
        #expect(sink.tokenIDs() == [[1]])
        #expect(sink.finishReasons() == [.maximumTokenCount])
        #expect(state.generatedText == "hel")
    }

    private static func handler(
        state: GenerationState,
        sink: RecordingBatchSink,
        decodeToken: @escaping @Sendable (Int) -> String
    ) -> MLXContinuousBatchTokenHandler {
        let clock = ContinuousClock()
        return MLXContinuousBatchTokenHandler(
            state: state,
            sink: sink.streamSink(),
            decodeToken: decodeToken
        ) { clock.now }
    }
}
