import Foundation
@testable import MLXLocalModels
import Testing

enum MLXTipCompactionSupport {
    static func expectCompacted(
        _ first: MLXPersistentPromptCacheBlockRecord?,
        _ second: MLXPersistentPromptCacheBlockRecord?,
        fixture: BlockFixture,
        payload: Data
    ) throws {
        #expect(first == nil)
        let compacted = try #require(second)
        #expect(compacted.blockHash == fixture.records[0].blockHash)
        #expect(compacted.payloadKind == .compactedRotatingTip)
        #expect(compacted.byteCount == payload.count)
        try Self.expectStoredKinds(fixture: fixture)
        try Self.expectBlockPayload(compacted, root: fixture.root, equals: payload)
    }

    static func expectStoredKinds(fixture: BlockFixture) throws {
        let compacted = try Self.record(
            fixture.records[0],
            root: fixture.root,
            payloadKinds: [.compactedRotatingTip]
        )
        #expect(compacted?.payloadKind == .compactedRotatingTip)
        let fullFirst = try Self.record(
            fixture.records[0],
            root: fixture.root,
            payloadKinds: [.block]
        )
        #expect(fullFirst == nil)
        try Self.expectBlockPayload(
            fixture.records[1],
            root: fixture.root,
            equals: Data(repeating: 2, count: 60)
        )
        try Self.expectBlockPayload(
            fixture.records[2],
            root: fixture.root,
            equals: Data(repeating: 3, count: 70)
        )
    }

    static func expectSegmentKinds(
        hashes: [String],
        signature: PromptCacheSignature,
        root: URL
    ) throws {
        let first = try Self.segmentRecord(
            hashes[0],
            signature: signature,
            root: root,
            kind: .compactedRotatingTip
        )
        let second = try Self.segmentRecord(
            hashes[1],
            signature: signature,
            root: root,
            kind: .compactedRotatingTip
        )
        let third = try Self.segmentRecord(hashes[2], signature: signature, root: root, kind: .block)
        let fourth = try Self.segmentRecord(hashes[3], signature: signature, root: root, kind: .block)

        #expect(first != nil)
        #expect(second != nil)
        #expect(third != nil)
        #expect(fourth != nil)
    }

    static func storeThreeBlocks(
        signature: PromptCacheSignature
    ) throws -> BlockFixture {
        let root = try Self.makeTemporaryDirectory()
        let records = try [
            Self.store(hash: Self.hash("a"), bytes: 50, value: 1, signature: signature, root: root),
            Self.store(hash: Self.hash("b"), bytes: 60, value: 2, signature: signature, root: root),
            Self.store(hash: Self.hash("c"), bytes: 70, value: 3, signature: signature, root: root)
        ]
        return BlockFixture(root: root, records: records)
    }

    static func recordExtension(
        _ previous: MLXPersistentPromptCacheBlockRecord,
        _ newest: MLXPersistentPromptCacheBlockRecord,
        root: URL,
        payload: Data
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try MLXPersistentPromptCacheBlockStore.recordRotatingTipExtension(
            previousTip: Self.descriptor(previous, root: root),
            newTip: Self.descriptor(newest, root: root),
            compactedPayload: payload
        )
    }

    static func storeSegments(
        tokens: [Int],
        signature: PromptCacheSignature,
        root: URL
    ) throws {
        try MLXPersistentPromptCacheSegmentStore.storeSegments(
            entry: PromptCacheEntry(tokens: tokens, cache: [], signature: signature, byteCount: 0),
            blockSize: 4,
            rootURL: root,
            encoder: Self.encodeSegment
        )
    }

    static func store(
        hash: String,
        bytes: Int,
        value: UInt8,
        signature: PromptCacheSignature,
        root: URL
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            tokenCount: 4,
            signature: signature,
            payload: Data(repeating: value, count: bytes),
            payloadKind: .block,
            rootURL: root
        )
    }

    static func record(
        _ record: MLXPersistentPromptCacheBlockRecord,
        root: URL,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind]
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try MLXPersistentPromptCacheBlockStore.storedRecord(
            blockHash: record.blockHash,
            signature: record.signature,
            blockSize: record.blockSize,
            rootURL: root,
            payloadKinds: payloadKinds
        )
    }

    static func segmentRecord(
        _ hash: String,
        signature: PromptCacheSignature,
        root: URL,
        kind: MLXPersistentPromptCacheBlockPayloadKind
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try MLXPersistentPromptCacheBlockStore.storedRecord(
            blockHash: hash,
            signature: signature,
            blockSize: 4,
            rootURL: root,
            payloadKinds: [kind]
        )
    }

    static func descriptor(
        _ record: MLXPersistentPromptCacheBlockRecord,
        root: URL
    ) -> MLXPersistentPromptCacheTipDescriptor {
        MLXPersistentPromptCacheTipDescriptor(
            blockHash: record.blockHash,
            blockSize: record.blockSize,
            signature: record.signature,
            rootURL: root
        )
    }

    static func expectBlockPayload(
        _ record: MLXPersistentPromptCacheBlockRecord,
        root: URL,
        equals expected: Data
    ) throws {
        let payload = try Data(contentsOf: MLXPersistentPromptCacheBlockStore.dataURL(
            for: record,
            rootURL: root
        ))
        #expect(payload == expected)
    }

    static func rotatingSignature() -> PromptCacheSignature {
        PromptCacheSignature(
            parameters: GenerateParameters(),
            cacheLayout: ["main[0]:RotatingKVCache(keep:0,maxSize:4096,step:1)"]
        )
    }

    static func kvSignature() -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(), cacheLayout: ["main[0]:KVCache"])
    }

    static func encodeSegment(_: [KVCache], _ metadata: [String: String]) throws -> Data {
        try JSONEncoder().encode(metadata)
    }

    static func hash(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    static func withHotCache<T>(
        limitBytes: Int,
        _ operation: () async throws -> T
    ) async throws -> T {
        try await PromptCacheTestIsolation.withLock {
            PromptCacheTestIsolation.resetSharedHotCache()
            MLXPersistentPromptCacheBlockStore.configureHotCache(limitBytes: limitBytes)
            defer { PromptCacheTestIsolation.resetSharedHotCache() }
            return try await operation()
        }
    }

    struct BlockFixture {
        let root: URL
        let records: [MLXPersistentPromptCacheBlockRecord]
    }
}
