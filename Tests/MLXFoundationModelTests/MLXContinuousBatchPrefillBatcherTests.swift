@testable import MLXLocalModels
import Testing

@Suite("MLX continuous batch prefill batcher")
struct MLXContinuousBatchPrefillBatcherTests {
    @Test("partitions requests by prompt length")
    func partitionsRequestsByPromptLength() throws {
        let groups = try MLXContinuousBatchPrefillBatcher.groups(for: [
            Self.request(tokens: [1, 2]),
            Self.request(tokens: [3]),
            Self.request(tokens: [4, 5])
        ])

        #expect(groups.map { $0.map(\.promptTokenIDs) } == [
            [[1, 2], [4, 5]],
            [[3]]
        ])
    }

    @Test("partitions requests by cache-affecting generation parameters")
    func partitionsRequestsByCacheAffectingParameters() throws {
        let groups = try MLXContinuousBatchPrefillBatcher.groups(for: [
            Self.request(tokens: [1, 2], parameters: GenerateParameters(maxTokens: 4)),
            Self.request(
                tokens: [3, 4],
                parameters: GenerateParameters(maxTokens: 4, maxKVSize: 32)
            ),
            Self.request(tokens: [5, 6], parameters: GenerateParameters(maxTokens: 8))
        ])

        #expect(groups.map { $0.map(\.promptTokenIDs) } == [
            [[1, 2], [5, 6]],
            [[3, 4]]
        ])
    }

    @Test("groups mergeable cached-prefix requests by cache layout")
    func groupsMergeableCachedPrefixRequestsByCacheLayout() throws {
        let groups = try MLXContinuousBatchPrefillBatcher.groups(for: [
            Self.request(tokens: [2], prefixCache: Self.mergeablePrefixCache()),
            Self.request(tokens: [4], prefixCache: Self.mergeablePrefixCache())
        ])

        #expect(groups.map { $0.map(\.promptTokenIDs) } == [
            [[2], [4]]
        ])
    }

    @Test("isolates unsupported cached-prefix requests into singleton groups")
    func isolatesUnsupportedCachedPrefixRequests() throws {
        let groups = try MLXContinuousBatchPrefillBatcher.groups(for: [
            Self.request(tokens: [2], prefixCache: Self.unsupportedPrefixCache()),
            Self.request(tokens: [4], prefixCache: Self.unsupportedPrefixCache())
        ])

        #expect(groups.map { $0.map(\.promptTokenIDs) } == [
            [[2]],
            [[4]]
        ])
    }

    @Test("rejects empty prompts before touching MLX arrays")
    func rejectsEmptyPromptsBeforeArrays() throws {
        do {
            _ = try MLXContinuousBatchPrefillBatcher.groups(for: [
                Self.request(tokens: [1, 2]),
                Self.request(tokens: [])
            ])
            Issue.record("Expected empty-prompt rejection")
        } catch MLXContinuousBatchPrefillError.emptyPrompt(let rowIndex) {
            #expect(rowIndex == 1)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    private static func request(
        tokens: [Int],
        parameters: GenerateParameters = GenerateParameters(maxTokens: 3),
        prefixCache: MLXContinuousBatchPrefixCache? = nil
    ) -> MLXContinuousBatchPrefillRequest {
        MLXContinuousBatchPrefillRequest(
            promptTokenIDs: tokens,
            parameters: parameters,
            tokenText: { "token-\($0)" },
            sink: RecordingBatchSink().streamSink(),
            prefixCache: prefixCache
        )
    }

    private static func mergeablePrefixCache() -> MLXContinuousBatchPrefixCache {
        MLXContinuousBatchPrefixCache(caches: [KVCacheSimple()], cachedTokenCount: 1)
    }

    private static func unsupportedPrefixCache() -> MLXContinuousBatchPrefixCache {
        MLXContinuousBatchPrefixCache(
            caches: [PromptCacheTestCache(offset: 1)],
            cachedTokenCount: 1
        )
    }
}
