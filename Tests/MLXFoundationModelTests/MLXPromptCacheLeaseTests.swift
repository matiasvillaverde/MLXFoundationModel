import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX prompt cache leases")
struct MLXPromptCacheLeaseTests {
    @Test("acquires a lease for prompt cache entries")
    func acquiresLeaseForPromptCacheEntries() throws {
        let signature = Self.signature()
        var entries = [
            Self.entry(tokens: [1, 2], signature: signature, byteCount: 10)
        ]

        let lease = try #require(PromptCachePlanner.acquireLease(at: 0, in: &entries))

        #expect(lease.signature == signature)
        #expect(lease.tokens == [1, 2])
        #expect(entries[0].activeLeaseIDs == [lease.id])
    }

    @Test("store does not prune leased entries to satisfy count limits")
    func storeDoesNotPruneLeasedEntriesToSatisfyCountLimits() {
        let signature = Self.signature()
        let lease = Self.lease(tokens: [1], signature: signature)
        var leased = Self.entry(tokens: [1], signature: signature, byteCount: 10)
        leased.activeLeaseIDs.insert(lease.id)
        var entries = [leased]

        PromptCachePlanner.store(
            Self.entry(tokens: [2], signature: signature, byteCount: 10),
            in: &entries,
            maxBytes: nil,
            maxEntries: 1
        )

        #expect(entries.map(\.tokens) == [[1]])
        #expect(entries.first?.activeLeaseIDs == [lease.id])
    }

    @Test("store does not replace leased duplicate entries")
    func storeDoesNotReplaceLeasedDuplicateEntries() {
        let signature = Self.signature()
        let lease = Self.lease(tokens: [1, 2], signature: signature)
        var leased = Self.entry(tokens: [1, 2], signature: signature, byteCount: 10)
        leased.activeLeaseIDs.insert(lease.id)
        var entries = [leased]

        PromptCachePlanner.store(
            Self.entry(tokens: [1, 2], signature: signature, byteCount: 5),
            in: &entries,
            maxBytes: nil,
            maxEntries: 2
        )

        #expect(entries.count == 2)
        #expect(entries.contains { $0.activeLeaseIDs == [lease.id] })
    }

    @Test("released entries can be pruned")
    func releasedEntriesCanBePruned() throws {
        let signature = Self.signature()
        let lease = Self.lease(tokens: [1], signature: signature)
        var leased = Self.entry(tokens: [1], signature: signature, byteCount: 10)
        leased.activeLeaseIDs.insert(lease.id)
        var entries = [leased]

        PromptCachePlanner.release(lease, in: &entries)
        PromptCachePlanner.store(
            Self.entry(tokens: [2], signature: signature, byteCount: 10),
            in: &entries,
            maxBytes: nil,
            maxEntries: 1
        )

        #expect(entries.map(\.tokens) == [[2]])
        #expect(try #require(entries.first).activeLeaseIDs.isEmpty)
    }

    @Test("cache layout is part of prompt cache compatibility")
    func cacheLayoutIsPartOfPromptCacheCompatibility() {
        let simpleSignature = Self.signature(cacheLayout: ["main[0]:KVCache"])
        let rotatingSignature = Self.signature(
            cacheLayout: ["main[0]:RotatingKVCache(keep:0,maxSize:128,step:256)"]
        )
        var entries = [
            Self.entry(tokens: [1, 2], signature: simpleSignature, byteCount: 10)
        ]

        PromptCachePlanner.store(
            Self.entry(tokens: [1, 2], signature: rotatingSignature, byteCount: 10),
            in: &entries,
            maxBytes: nil,
            maxEntries: 4
        )

        #expect(entries.map(\.signature.cacheLayout) == [
            rotatingSignature.cacheLayout,
            simpleSignature.cacheLayout
        ])
    }

    @Test("cache layout fingerprints nested and draft caches")
    func cacheLayoutFingerprintsNestedAndDraftCaches() {
        let layout = PromptCachePlanner.cacheLayoutFingerprint(
            for: [
                CacheList(
                    KVCacheSimple(),
                    RotatingKVCache(maxSize: 64, keep: 8)
                )
            ],
            draftCache: [
                QuantizedKVCache(groupSize: 32, bits: 4)
            ]
        )

        #expect(layout == [
            "main[0]:CacheList[KVCache,RotatingKVCache(keep:8,maxSize:64,step:256)]",
            "draft[0]:QuantizedKVCache(groupSize:32,bits:4,mode:affine)"
        ])
    }

    @Test("make entry trims generated tokens from cache snapshots")
    func makeEntryTrimsGeneratedTokensFromCacheSnapshots() throws {
        let cache = Self.cache(offset: 6)

        let entry = try #require(PromptCachePlanner.makeEntry(
            tokenIds: [1, 2, 3, 4],
            cache: [cache],
            parameters: GenerateParameters(),
            maxBytes: nil
        ))
        let storedCache = try #require(entry.cache.first)

        #expect(cache.offset == 6)
        #expect(storedCache.offset == 4)
    }

    @Test("cache list trim trims every child cache")
    func cacheListTrimTrimsEveryChildCache() {
        let first = Self.cache(offset: 6)
        let second = Self.cache(offset: 6)
        let cacheList = CacheList(first, second)

        let trimmed = cacheList.trim(2)

        #expect(trimmed == 2)
        #expect(first.offset == 4)
        #expect(second.offset == 4)
    }

    private static func entry(
        tokens: [Int],
        signature: PromptCacheSignature,
        byteCount: Int
    ) -> PromptCacheEntry {
        PromptCacheEntry(
            tokens: tokens,
            cache: [],
            signature: signature,
            byteCount: byteCount
        )
    }

    private static func lease(
        tokens: [Int],
        signature: PromptCacheSignature
    ) -> PromptCacheLease {
        PromptCacheLease(
            id: UUID(),
            signature: signature,
            tokens: tokens
        )
    }

    private static func signature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters())
    }

    private static func signature(cacheLayout: [String]) -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(), cacheLayout: cacheLayout)
    }

    private static func cache(offset: Int) -> PromptCacheTestCache {
        PromptCacheTestCache(offset: offset)
    }
}
