import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX continuous batch prompt cache storage",
    .disabled(
        if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
        "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
    )
)
struct MLXContinuousBatchCacheStorageTests {
    @Test("stores one prompt-cache entry from a batched prefill cache row")
    func storesOnePromptCacheEntryFromBatchedPrefillCacheRow() throws {
        let identity = PromptCacheIdentity(stableFingerprint: "continuous-batch-storage-test")
        var entries: [PromptCacheEntry] = []

        try Device.withDefaultDevice(.cpu) {
            try Self.storage(identity: identity).store(
                cache: [Self.batchedCache()],
                rowIndex: 1,
                rowCount: 2,
                entries: &entries
            )
        }

        #expect(entries.count == 1)
        let entry = try #require(entries.first)
        #expect(entry.tokens == [10, 11, 12])
        #expect(entry.cache[0].state[0].shape == [1, 1, 3, 2])
        #expect(entry.cache[0].state[0].asArray(Float.self) == [7, 8, 9, 10, 11, 12])
    }

    @Test("stored continuous-batch prompt cache can be planned for reuse")
    func storedContinuousBatchPromptCacheCanBePlannedForReuse() throws {
        let identity = PromptCacheIdentity(stableFingerprint: "continuous-batch-plan-test")
        var entries: [PromptCacheEntry] = []

        try Device.withDefaultDevice(.cpu) {
            try Self.storage(identity: identity).store(
                cache: [Self.batchedCache()],
                rowIndex: 0,
                rowCount: 2,
                entries: &entries
            )
        }

        let plan = Self.reusePlan(identity: identity, entries: &entries)
        #expect(plan.reusedTokenCount == 3)
        #expect(plan.input.text.tokens.asArray(Int.self) == [13])
        #expect(plan.cache?[0].state[0].shape == [1, 1, 3, 2])
    }

    private static func storage(
        identity: PromptCacheIdentity
    ) -> MLXContinuousBatchPromptCacheStorage {
        MLXContinuousBatchPromptCacheStorage(
            tokenIDs: [10, 11, 12],
            request: MLXPromptCacheEntryStore.Request(
                parameters: GenerateParameters(maxTokens: 2),
                cacheVariant: nil,
                promptCacheIdentity: identity,
                maxBytes: nil,
                reusePromptCache: true,
                runtimePreferences: ModelRuntimePreferences(promptCachePolicy: .memory)
            )
        )
    }

    private static func reusePlan(
        identity: PromptCacheIdentity,
        entries: inout [PromptCacheEntry]
    ) -> PromptCachePlan {
        let cacheLayout = entries.first.map { entry in
            PromptCachePlanner.cacheLayoutFingerprint(for: entry.cache)
        }
        return PromptCachePlanner.plan(
            fullInput: LMInput(tokens: MLXArray([10, 11, 12, 13])),
            tokenIds: [10, 11, 12, 13],
            parameters: GenerateParameters(maxTokens: 2),
            cacheLayout: cacheLayout,
            promptCacheIdentity: identity,
            existingEntries: &entries,
            reuseEnabled: true
        )
    }

    private static func batchedCache() -> KVCache {
        let cache = KVCacheSimple()
        cache.state = [
            MLXArray(Self.values(start: 1, count: 12)).reshaped([2, 1, 3, 2]),
            MLXArray(Self.values(start: 13, count: 12)).reshaped([2, 1, 3, 2])
        ]
        return cache
    }

    private static func values(start: Float, count: Int) -> [Float] {
        (0 ..< count).map { start + Float($0) }
    }
}
