import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX prompt cache hot payload capacity", .serialized)
struct MLXPromptCacheHotPayloadCapacityTests {
    @Test("store-if-fits refuses promotion without evicting hot payloads")
    func storeIfFitsRefusesPromotionWithoutEvictingHotPayloads() {
        let cache = MLXPersistentPromptCacheHotPayloadCache(maxBytes: 32)

        for index in 1 ... 4 {
            let data = Data(repeating: UInt8(index), count: 8)
            #expect(cache.storeIfFits(data, forKey: "\(index)"))
        }

        #expect(!cache.storeIfFits(Data(repeating: 5, count: 8), forKey: "5"))
        let cappedSnapshot = cache.snapshot()

        #expect(cappedSnapshot.entryCount == 4)
        #expect(cappedSnapshot.totalBytes == 32)
        #expect(cappedSnapshot.keys == ["1", "2", "3", "4"])

        let evictedCount = cache.store(Data(repeating: 5, count: 8), forKey: "5")
        let evictingSnapshot = cache.snapshot()

        #expect(evictedCount == 1)
        #expect(evictingSnapshot.entryCount == 4)
        #expect(evictingSnapshot.totalBytes == 32)
        #expect(evictingSnapshot.keys == ["2", "3", "4", "5"])
    }

    @Test("store-if-fits keeps existing entry when replacement is too large")
    func storeIfFitsKeepsExistingEntryWhenReplacementIsTooLarge() throws {
        let cache = MLXPersistentPromptCacheHotPayloadCache(maxBytes: 8)

        #expect(cache.storeIfFits(Data(repeating: 1, count: 4), forKey: "key"))
        #expect(!cache.storeIfFits(Data(repeating: 2, count: 9), forKey: "key"))

        let data = try #require(cache.data(forKey: "key"))
        let snapshot = cache.snapshot()
        #expect(data == Data(repeating: 1, count: 4))
        #expect(snapshot.entryCount == 1)
        #expect(snapshot.totalBytes == 4)
    }

    @Test("segment restore caps disk loads by available hot payload capacity")
    func segmentRestoreCapsDiskLoadsByAvailableHotPayloadCapacity() async throws {
        try await PromptCacheTestIsolation.withLock {
            let root = try Self.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            defer { PromptCacheTestIsolation.resetSharedHotCache() }

            PromptCacheTestIsolation.resetSharedHotCache()
            let records = try Self.storeSegmentFixtures(root: root)
            let byteLimit = records.prefix(2).reduce(0) { total, record in
                total + record.byteCount
            }
            MLXPersistentPromptCacheBlockStore.clearHotCache()
            MLXPersistentPromptCacheBlockStore.configureHotCache(limitBytes: byteLimit)

            var loadedURLs: [URL] = []
            let segments = try MLXPersistentPromptCacheSegmentStore.restorePrefixSegments(
                tokenIds: Array(0 ..< 12),
                signature: Self.signature(),
                blockSize: Self.blockSize,
                rootURL: root
            ) { url in
                loadedURLs.append(url)
                let data = try MLXPersistentPromptCacheBlockStore.loadPayload(
                    at: url,
                    promotionPolicy: .skipIfWouldEvict
                )
                return ([], try JSONDecoder().decode([String: String].self, from: data))
            }

            #expect(segments.map(\.blockIndex) == [0, 1])
            #expect(segments.flatMap(\.tokens) == Array(0 ..< 8))
            #expect(loadedURLs.count == 2)
            #expect(MLXPersistentPromptCacheBlockStore.hotCacheSnapshot().totalBytes == byteLimit)
        }
    }

    private static let blockSize = 4

    private static func storeSegmentFixtures(
        root: URL
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        try MLXPersistentPromptCacheSegmentStore.storeSegments(
            entry: PromptCacheEntry(
                tokens: Array(0 ..< 12),
                cache: [],
                signature: signature(),
                byteCount: 0
            ),
            blockSize: blockSize,
            rootURL: root,
            encoder: encode
        )
    }

    private static func encode(
        _: [KVCache],
        _ metadata: [String: String]
    ) throws -> Data {
        try JSONEncoder().encode(metadata)
    }

    private static func signature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters())
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}
