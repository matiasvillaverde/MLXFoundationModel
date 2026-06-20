import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MiniMax M3 KV cache")
struct MiniMaxM3KVCacheTests {
    @Test("single sparse cache flattens state and trims index state")
    func singleSparseCacheFlattensStateAndTrimsIndexState() {
        Device.withDefaultDevice(.cpu) {
            let cache = Self.singleCache(length: 8, keyStart: 1, indexStart: 100)

            #expect(cache.state.count == 3)
            #expect(cache.state[0].shape == [1, 2, 8, 3])
            #expect(cache.state[2].shape == [1, 1, 8, 3])
            #expect(cache.metaState == ["8"])

            #expect(cache.trim(3) == 3)
            #expect(cache.offset == 5)
            #expect(cache.metaState == ["5"])
            #expect(cache.state[0].shape == [1, 2, 5, 3])
            #expect(cache.state[2].shape == [1, 1, 5, 3])
        }
    }

    @Test("single sparse cache copy preserves index keys")
    func singleSparseCacheCopyPreservesIndexKeys() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = Self.singleCache(length: 4, keyStart: 1, indexStart: 100)
            let copied = try #require(cache.copy() as? MiniMaxM3KVCache)

            #expect(copied.state.map(\.shape) == cache.state.map(\.shape))
            #expect(copied.metaState == cache.metaState)
            #expect(copied.state[2].asArray(Float.self) == cache.state[2].asArray(Float.self))
        }
    }

    @Test("batch cache merge and extract preserve left padded sparse index keys")
    func batchCacheMergeAndExtractPreserveLeftPaddedSparseIndexKeys() {
        Device.withDefaultDevice(.cpu) {
            let first = Self.singleCache(length: 4, keyStart: 1, indexStart: 100)
            let second = Self.singleCache(length: 6, keyStart: 10, indexStart: 200)
            let batch = MiniMaxM3BatchKVCache.merge([first, second])

            #expect(!batch.isTrimmable)
            #expect(batch.state[0].shape == [2, 2, 6, 3])
            #expect(Self.intValues(batch.state[2]) == [4, 6])
            #expect(Self.intValues(batch.state[3]) == [2, 0])
            #expect(batch.metaState == ["6"])

            let extracted = batch.extract(0)
            #expect(extracted.state[0].shape == [1, 2, 4, 3])
            #expect(extracted.state[2].shape == [1, 1, 4, 3])
            #expect(extracted.state[0].asArray(Float.self).first == 1)
            #expect(extracted.state[2].asArray(Float.self).first == 100)
        }
    }

    @Test("single sparse cache round trips through prompt cache serialization")
    func singleSparseCacheRoundTripsThroughPromptCacheSerialization() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = Self.singleCache(length: 4, keyStart: 1, indexStart: 100)
            let loaded = try Self.roundTrip([cache])
            let loadedCache = try #require(loaded.first as? MiniMaxM3KVCache)

            #expect(PromptCachePlanner.cacheLayoutFingerprint(for: loaded) == [
                "main[0]:MiniMaxM3KVCache(indexKeys:true)"
            ])
            #expect(loadedCache.state.map(\.shape) == cache.state.map(\.shape))
            #expect(loadedCache.metaState == ["4"])
        }
    }

    @Test("batch sparse cache round trips through prompt cache serialization")
    func batchSparseCacheRoundTripsThroughPromptCacheSerialization() throws {
        try Device.withDefaultDevice(.cpu) {
            let first = Self.singleCache(length: 4, keyStart: 1, indexStart: 100)
            let second = Self.singleCache(length: 6, keyStart: 10, indexStart: 200)
            let batch = MiniMaxM3BatchKVCache.merge([first, second])
            let loaded = try Self.roundTrip([batch])
            let loadedBatch = try #require(loaded.first as? MiniMaxM3BatchKVCache)

            #expect(PromptCachePlanner.cacheLayoutFingerprint(for: loaded) == [
                "main[0]:MiniMaxM3BatchKVCache(fullState:true)"
            ])
            #expect(Self.intValues(loadedBatch.state[2]) == [4, 6])
            #expect(Self.intValues(loadedBatch.state[3]) == [2, 0])
            #expect(loadedBatch.extract(1).state[2].shape == [1, 1, 6, 3])
        }
    }

    private static func singleCache(
        length: Int,
        keyStart: Float,
        indexStart: Float
    ) -> MiniMaxM3KVCache {
        let cache = MiniMaxM3KVCache()
        cache.state = [
            array(length: length, heads: 2, start: keyStart),
            array(length: length, heads: 2, start: keyStart + 1_000),
            array(length: length, heads: 1, start: indexStart)
        ]
        return cache
    }

    private static func array(length: Int, heads: Int, start: Float) -> MLXArray {
        MLXArray((0 ..< heads * length * 3).map { start + Float($0) })
            .reshaped([1, heads, length, 3])
    }

    private static func intValues(_ array: MLXArray) -> [Int] {
        eval(array)
        return array.asArray(Int32.self).map(Int.init)
    }

    private static func roundTrip(_ cache: [KVCache]) throws -> [KVCache] {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("cache.safetensors")
        try savePromptCache(url: url, cache: cache)
        let (loaded, _) = try loadPromptCache(url: url)
        return loaded
    }

    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        return directory
    }
}
