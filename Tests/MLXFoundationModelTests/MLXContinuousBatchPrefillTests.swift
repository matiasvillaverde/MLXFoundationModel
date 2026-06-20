import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch prefill")
struct MLXContinuousBatchPrefillTests {
    @Test("rejects mismatched prompt lengths before touching MLX arrays")
    func rejectsMismatchedPromptLengths() throws {
        #expect(throws: MLXContinuousBatchPrefillError.self) {
            try MLXContinuousBatchPrefill.run(
                model: MLXEchoBatchLanguageModel(),
                requests: [
                    Self.request(tokens: [1, 2]),
                    Self.request(tokens: [3])
                ]
            )
        }
    }

    @Test("rejects cache-incompatible generation parameters before batching")
    func rejectsCacheIncompatibleParameters() throws {
        #expect(throws: MLXContinuousBatchPrefillError.self) {
            try MLXContinuousBatchPrefill.run(
                model: MLXEchoBatchLanguageModel(),
                requests: [
                    Self.request(tokens: [1, 2]),
                    Self.request(
                        tokens: [3, 4],
                        parameters: GenerateParameters(maxTokens: 3, maxKVSize: 16, temperature: 0)
                    )
                ]
            )
        }
    }

    @Test(
        "samples first tokens and returns active scheduler rows",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func samplesFirstTokensAndReturnsActiveSchedulerRows() throws {
        let firstSink = RecordingBatchSink()
        let secondSink = RecordingBatchSink()

        let result = try Device.withDefaultDevice(.cpu) {
            try MLXContinuousBatchPrefill.run(
                model: MLXEchoBatchLanguageModel(),
                requests: [
                    Self.request(tokens: [1, 2], sink: firstSink),
                    Self.request(tokens: [3, 4], sink: secondSink)
                ]
            )
        }

        #expect(result.firstTokenIDs == [1, 0])
        #expect(result.batch.orderedRowIDs == [0, 1])
        #expect(result.batch.coordinator[0]?.previousTokenID == 1)
        #expect(result.batch.coordinator[1]?.previousTokenID == 0)
        #expect(result.batch.coordinator[0]?.generatedTokenCount == 1)
        #expect(firstSink.texts() == ["token-1"])
        #expect(secondSink.texts() == ["token-0"])
        #expect(firstSink.finishReasons().isEmpty)
        #expect(secondSink.finishReasons().isEmpty)
    }

    @Test(
        "finishes rows completed by the first sampled token",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func finishesRowsCompletedByFirstSampledToken() throws {
        let sink = RecordingBatchSink()

        let result = try Device.withDefaultDevice(.cpu) {
            try MLXContinuousBatchPrefill.run(
                model: MLXEchoBatchLanguageModel(),
                requests: [
                    Self.request(
                        tokens: [1, 2],
                        parameters: GenerateParameters(maxTokens: 1, temperature: 0),
                        sink: sink
                    )
                ]
            )
        }

        #expect(result.firstTokenIDs == [1])
        #expect(result.batch.isEmpty)
        #expect(sink.texts() == ["token-1"])
        #expect(sink.finishReasons() == [.maximumTokenCount])
    }

    @Test(
        "merges cached prefix KV rows before suffix prefill",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func mergesCachedPrefixKVRowsBeforeSuffixPrefill() throws {
        let result = try Device.withDefaultDevice(.cpu) {
            try MLXContinuousBatchPrefill.run(
                model: MLXEchoBatchLanguageModel(),
                requests: [
                    Self.request(tokens: [2], prefixCache: Self.prefixCache(keySeed: 1)),
                    Self.request(tokens: [4], prefixCache: Self.prefixCache(keySeed: 5))
                ]
            )
        }

        let state = result.cache[0].state
        #expect(state[0].shape == [2, 1, 1, 2])
        #expect(state[1].shape == [2, 1, 1, 2])
        #expect(state[0].asArray(Float.self) == [1, 2, 5, 6])
        #expect(state[1].asArray(Float.self) == [3, 4, 7, 8])
    }

    @Test(
        "rejects mismatched cached prefix shapes before model execution",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func rejectsMismatchedCachedPrefixShapes() throws {
        #expect(throws: MLXContinuousBatchPrefillError.self) {
            try Device.withDefaultDevice(.cpu) {
                try MLXContinuousBatchPrefill.run(
                    model: MLXEchoBatchLanguageModel(),
                    requests: [
                        Self.request(tokens: [2], prefixCache: Self.prefixCache(keySeed: 1)),
                        Self.request(tokens: [4], prefixCache: Self.prefixCache(keySeed: 5, width: 3))
                    ]
                )
            }
        }
    }

    private static func request(
        tokens: [Int],
        parameters: GenerateParameters = GenerateParameters(maxTokens: 3, temperature: 0),
        sink: RecordingBatchSink = RecordingBatchSink(),
        prefixCache: MLXContinuousBatchPrefixCache? = nil
    ) -> MLXContinuousBatchPrefillRequest {
        MLXContinuousBatchPrefillRequest(
            promptTokenIDs: tokens,
            parameters: parameters,
            tokenText: { "token-\($0)" },
            sink: sink.streamSink(),
            prefixCache: prefixCache
        )
    }

    private static func prefixCache(
        keySeed: Float,
        width: Int = 2
    ) -> MLXContinuousBatchPrefixCache {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray(Self.values(startingAt: keySeed, count: width)).reshaped([1, 1, 1, width]),
            MLXArray(Self.values(startingAt: keySeed + 2, count: width)).reshaped([1, 1, 1, width])
        ]
        return MLXContinuousBatchPrefixCache(caches: [cache], cachedTokenCount: 1)
    }

    private static func values(startingAt start: Float, count: Int) -> [Float] {
        (0 ..< count).map { start + Float($0) }
    }
}
