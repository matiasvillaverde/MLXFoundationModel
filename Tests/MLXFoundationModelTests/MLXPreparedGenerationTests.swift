import Foundation
import MLX
@testable import MLXLocalModels
import Testing
import Tokenizers

@Suite(
    "MLX prepared generation",
    .disabled(
        if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
        "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
    )
)
struct MLXPreparedGenerationTests {
    @Test("builds a continuous-batch prefill request from an uncached prompt")
    func buildsContinuousBatchPrefillRequestFromUncachedPrompt() throws {
        MLXGenerationDiagnostics.reset()
        let session = MLXSession()
        var promptCacheEntries: [PromptCacheEntry] = []
        let genContext = Self.generationContext(stopSequences: ["STOP"])

        let prepared = try session.prepareGeneration(
            genContext: genContext,
            promptCacheEntries: &promptCacheEntries,
            speculativeDecoding: nil,
            promptCacheVariant: nil
        )
        let sink = RecordingBatchSink()
        let request = try prepared.makeContinuousBatchPrefillRequest(sink: sink.streamSink())

        #expect(prepared.promptTokenIDs == [10, 11])
        #expect(request.promptTokenIDs == [10, 11])
        #expect(request.parameters.maxTokens == 2)
        #expect(request.stopTokenIDs == [0, 99, 100])
        #expect(promptCacheEntries.isEmpty)
        #expect(Self.promptCachePlanSnapshots() == [
            MLXPromptCachePlanSnapshot(promptTokenCount: 2, reusedTokenCount: 0)
        ])
    }

    @Test("uses shared stream semantics for prepared continuous-batch requests")
    func usesSharedStreamSemanticsForPreparedContinuousBatchRequests() throws {
        let prepared = Self.preparedGeneration(stopSequences: ["STOP"])
        let sink = RecordingBatchSink()
        let request = try prepared.makeContinuousBatchPrefillRequest(sink: sink.streamSink())

        #expect(request.stream(tokenID: 20) == .suppressed)
        #expect(request.stream(tokenID: 21) == .finish(.streamRequestedStop(21)))

        #expect(sink.texts().isEmpty)
        #expect(prepared.state.generatedTokenCount == 2)
        #expect(prepared.state.stopReason == .stopSequence)
    }

    @Test("builds cached-prefix continuous-batch requests with suffix prefill tokens")
    func buildsCachedPrefixContinuousBatchRequest() throws {
        let plan = PromptCachePlan(
            input: LMInput(tokens: MLXArray([11])),
            cache: [Self.simpleCache()],
            reusedTokenCount: 1
        )
        let prepared = Self.preparedGeneration(cachePlan: plan)
        let sink = RecordingBatchSink()

        let request = try prepared.makeContinuousBatchPrefillRequest(sink: sink.streamSink())

        #expect(request.promptTokenIDs == [11])
        #expect(request.processorPromptTokenIDs == [10, 11])
        #expect(request.prefixCache?.cachedTokenCount == 1)
        #expect(request.prefixCache?.supportsMultiRowMerge == true)
    }

    @Test("refuses speculative decoding until MTP batching is connected")
    func refusesSpeculativeDecodingUntilMTPBatchingIsConnected() {
        let prepared = Self.preparedGeneration(usesSpeculativeDecoding: true)

        Self.expectBatchingError(.speculativeDecodingUnsupported, from: prepared)
    }

    @Test("refuses empty prompts because rows need a previous token")
    func refusesEmptyPromptsBecauseRowsNeedPreviousToken() {
        let fullInput = LMInput(tokens: MLXArray([] as [Int]))
        let plan = PromptCachePlan(input: fullInput, cache: nil, reusedTokenCount: 0)
        let prepared = Self.preparedGeneration(
            fullInput: fullInput,
            promptTokenIDs: [],
            cachePlan: plan
        )

        Self.expectBatchingError(.emptyPrompt, from: prepared)
    }

    private static func promptCachePlanSnapshots() -> [MLXPromptCachePlanSnapshot] {
        MLXGenerationDiagnostics.events().compactMap { event in
            if case .promptCachePlan(let snapshot) = event {
                return snapshot
            }
            return nil
        }
    }

    private static func expectBatchingError(
        _ expectedError: MLXPreparedGenerationBatchingError,
        from prepared: MLXPreparedGeneration
    ) {
        do {
            _ = try prepared.makeContinuousBatchPrefillRequest(sink: RecordingBatchSink().streamSink())
            Issue.record("Expected \(expectedError)")
        } catch let error as MLXPreparedGenerationBatchingError {
            #expect(error == expectedError)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    private static func preparedGeneration(
        fullInput: LMInput = LMInput(tokens: MLXArray([10, 11])),
        promptTokenIDs: [Int] = [10, 11],
        cachePlan: PromptCachePlan? = nil,
        stopSequences: [String] = [],
        usesSpeculativeDecoding: Bool = false
    ) -> MLXPreparedGeneration {
        let genContext = generationContext(stopSequences: stopSequences)
        let state = GenerationState()
        let tokenContext = TokenContext(
            state: state,
            context: genContext.modelContext,
            input: genContext.input,
            continuation: genContext.continuation,
            clock: genContext.clock
        )
        state.stopDetector = StopSequenceDetector(sequences: stopSequences)
        state.detokenizer = NaiveStreamingDetokenizer(tokenizer: genContext.modelContext.tokenizer)
        return MLXPreparedGeneration(
            genContext: genContext,
            fullInput: fullInput,
            promptTokenIDs: promptTokenIDs,
            cachePlan: cachePlan ?? PromptCachePlan(
                input: fullInput,
                cache: nil,
                reusedTokenCount: 0
            ),
            state: state,
            tokenContext: tokenContext,
            promptStartTime: genContext.clock.now,
            usesSpeculativeDecoding: usesSpeculativeDecoding,
            adaptivePrefillController: nil
        )
    }

    private static func generationContext(
        stopSequences: [String] = []
    ) -> GenerationContext {
        let clock = ContinuousClock()
        return GenerationContext(
            modelContext: modelContext(),
            input: input(stopSequences: stopSequences),
            parameters: GenerateParameters(maxTokens: 2),
            generationStartTime: clock.now,
            continuation: streamContinuation(),
            clock: clock,
            runtimePreferences: .default,
            memoryProfile: nil
        )
    }

    private static func modelContext() -> ModelContext {
        ModelContext(
            configuration: ModelConfiguration(
                id: "test/prepared",
                extraEOSTokens: ["<extra>"],
                eosTokenIds: [99]
            ),
            model: MLXEchoBatchLanguageModel(),
            tokenizer: PreparedGenerationTokenizer()
        )
    }

    private static func input(stopSequences: [String]) -> LLMInput {
        LLMInput(
            context: "hello world",
            promptMetadata: PromptRenderMetadata(rendererID: "test"),
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: "prepared-test"),
            sampling: SamplingParameters(
                temperature: 0,
                topP: 1,
                topK: 1,
                stopSequences: stopSequences
            ),
            limits: ResourceLimits(maxTokens: 2, reusePromptCache: false)
        )
    }

    private static func simpleCache() -> KVCache {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray([Float(1), Float(2)]).reshaped([1, 1, 1, 2]),
            MLXArray([Float(3), Float(4)]).reshaped([1, 1, 1, 2])
        ]
        return cache
    }

    private static func streamContinuation() -> AsyncThrowingStream<LLMStreamChunk, Error>.Continuation {
        var continuation: AsyncThrowingStream<LLMStreamChunk, Error>.Continuation?
        _ = AsyncThrowingStream<LLMStreamChunk, Error> { streamContinuation in
            continuation = streamContinuation
        }
        guard let continuation else {
            preconditionFailure("AsyncThrowingStream did not provide a continuation")
        }
        return continuation
    }
}
