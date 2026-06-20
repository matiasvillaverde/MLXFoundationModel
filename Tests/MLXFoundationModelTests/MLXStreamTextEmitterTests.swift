@testable import MLXLocalModels
import Testing

@Suite("MLX stream text emitter")
struct MLXStreamTextEmitterTests {
    @Test("withholds stop-sequence suffixes and suppresses the stop marker")
    func withholdsStopSequenceSuffixes() async throws {
        let harness = Self.harness(stopSequences: ["STOP"])

        #expect(MLXStreamTextEmitter.append("hello ST", context: harness.context) == .more)
        #expect(harness.state.generatedText == "hello")
        #expect(MLXStreamTextEmitter.append("OP hidden", context: harness.context) == .stop)
        harness.continuation.finish()

        let chunks = try await Self.chunks(from: harness.stream)

        #expect(chunks.map(\.text) == ["hello", " "])
        #expect(harness.state.generatedText == "hello ")
        #expect(harness.state.stopReason == .stopSequence)
    }

    @Test("flush emits a pending non-stop suffix")
    func flushEmitsPendingNonStopSuffix() async throws {
        let harness = Self.harness(stopSequences: ["STOP"])

        #expect(MLXStreamTextEmitter.append("hel", context: harness.context) == .more)
        #expect(harness.state.generatedText.isEmpty)
        MLXStreamTextEmitter.flush(context: harness.context)
        harness.continuation.finish()

        let chunks = try await Self.chunks(from: harness.stream)

        #expect(chunks.map(\.text) == ["hel"])
        #expect(harness.state.generatedText == "hel")
    }

    @Test("emits token IDs for buffered text")
    func emitsTokenIDsForBufferedText() async throws {
        let harness = Self.harness(stopSequences: [])
        harness.state.allTokens = [101, 102]
        harness.state.generatedTokenCount = 2

        MLXStreamTextEmitter.yield("hello", context: harness.context)
        harness.continuation.finish()

        let chunks = try await Self.chunks(from: harness.stream)

        #expect(chunks.map(\.text) == ["hello"])
        #expect(chunks.map(\.tokenIDs) == [[101, 102]])
    }

    private static func harness(
        stopSequences: [String]
    ) -> TextEmitterHarness {
        let state = GenerationState()
        state.stopDetector = StopSequenceDetector(sequences: stopSequences)

        var continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation?
        let stream = AsyncThrowingStream<LLMStreamChunk, Error> { streamContinuation in
            continuation = streamContinuation
        }
        guard let continuation else {
            preconditionFailure("AsyncThrowingStream did not provide a continuation")
        }

        return TextEmitterHarness(
            context: .init(continuation: continuation, state: state),
            continuation: continuation,
            state: state,
            stream: stream
        )
    }

    private static func chunks(
        from stream: AsyncThrowingStream<LLMStreamChunk, Error>
    ) async throws -> [LLMStreamChunk] {
        var chunks: [LLMStreamChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }
        return chunks
    }
}
