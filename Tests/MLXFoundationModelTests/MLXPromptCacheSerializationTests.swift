import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite(
    "MLX prompt cache serialization",
    .disabled(
        if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
        "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
    )
)
struct MLXPromptCacheSerializationTests {
    @Test("round trips nested CacheList layout")
    func roundTripsNestedCacheListLayout() throws {
        try Device.withDefaultDevice(.cpu) {
            let directory = try Self.temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: directory) }
            let url = directory.appendingPathComponent("cache.safetensors")
            let nested = Self.nestedCacheList()
            let expectedLayout = PromptCachePlanner.cacheLayoutFingerprint(for: [nested])

            try savePromptCache(
                url: url,
                cache: [nested],
                metadata: ["purpose": "cache-list-roundtrip"]
            )

            let (loadedCaches, metadata) = try loadPromptCache(url: url)

            #expect(metadata?["purpose"] == "cache-list-roundtrip")
            #expect(PromptCachePlanner.cacheLayoutFingerprint(for: loadedCaches) == expectedLayout)
            try Self.expectLoadedCacheList(loadedCaches, original: nested)
        }
    }

    private static func nestedCacheList() -> CacheList {
        let simple = KVCacheSimple()
        simple.state = state(start: 1)

        let rotating = RotatingKVCache(maxSize: 64, keep: 8)
        rotating.state = state(start: 11)
        rotating.metaState = ["8", "64", "256", "2", "2"]

        let chunked = ChunkedKVCache(chunkSize: 4)
        chunked.state = state(start: 21)

        return CacheList(simple, CacheList(rotating, chunked))
    }

    private static func expectLoadedCacheList(
        _ loadedCaches: [KVCache],
        original: CacheList
    ) throws {
        let loadedTop = try #require(loadedCaches.first as? CacheList)
        #expect(loadedTop.layoutCaches.count == 2)
        #expect(loadedTop.layoutCaches[0] is KVCacheSimple)

        let loadedNested = try #require(loadedTop.layoutCaches[1] as? CacheList)
        #expect(loadedNested.layoutCaches.count == 2)
        #expect(loadedNested.layoutCaches[0] is RotatingKVCache)
        #expect(loadedNested.layoutCaches[1] is ChunkedKVCache)
        #expect(loadedNested.layoutCaches[0].offset == 2)
        #expect(loadedNested.layoutCaches[1].metaState == ["4", "0"])
        #expect(loadedTop.state.map(\.shape) == original.state.map(\.shape))
        #expect(loadedTop.layoutCaches[0].state[0].asArray(Float.self) == [1, 2])
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

    private static func state(start: Float) -> [MLXArray] {
        [
            MLXArray([start, start + 1]).reshaped([1, 1, 2, 1]),
            MLXArray([start + 2, start + 3]).reshaped([1, 1, 2, 1])
        ]
    }
}
