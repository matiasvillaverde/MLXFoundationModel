import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache block store namespacing")
struct MLXBlockStoreNamespaceTests {
    @Test("same logical block stores separate signature-specific payloads")
    func sameLogicalBlockStoresSeparateSignatureSpecificPayloads() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = try Self.singleBlockFixture()
        let firstSignature = Self.signature(prefillStepSize: 128)
        let secondSignature = Self.signature(prefillStepSize: 256)

        let firstRecord = try Self.store(
            "first", fixture: fixture, signature: firstSignature, root: root, kind: .generic
        )
        let secondRecord = try Self.store(
            "second", fixture: fixture, signature: secondSignature, root: root, kind: .generic
        )
        let firstData = try Self.prefixData(
            tokens: fixture.tokenIds,
            signature: firstSignature,
            root: root
        )
        let secondData = try Self.prefixData(
            tokens: fixture.tokenIds,
            signature: secondSignature,
            root: root
        )

        #expect(firstRecord.blockHash == secondRecord.blockHash)
        #expect(firstRecord.storageHash != secondRecord.storageHash)
        #expect(firstData == Data("first".utf8))
        #expect(secondData == Data("second".utf8))
        #expect(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root).count == 2)
    }

    @Test("same logical block stores separate payload kinds")
    func sameLogicalBlockStoresSeparatePayloadKinds() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = try Self.singleBlockFixture()
        let signature = Self.signature()

        let blockRecord = try Self.store(
            "block", fixture: fixture, signature: signature, root: root, kind: .block
        )
        let snapshotRecord = try Self.store(
            "snapshot", fixture: fixture, signature: signature, root: root, kind: .prefixSnapshot
        )
        let blockData = try Self.prefixData(
            tokens: fixture.tokenIds,
            signature: signature,
            root: root
        )
        let snapshotData = try Self.snapshotData(
            tokens: fixture.tokenIds,
            signature: signature,
            root: root
        )

        #expect(blockRecord.blockHash == snapshotRecord.blockHash)
        #expect(blockRecord.storageHash != snapshotRecord.storageHash)
        #expect(blockData == Data("block".utf8))
        #expect(snapshotData == Data("snapshot".utf8))
        #expect(try MLXPersistentPromptCacheBlockStore.scan(rootURL: root).count == 2)
    }

    private static func singleBlockFixture() throws -> (tokenIds: [Int], hash: String) {
        let tokenIds = Array(0 ..< 4)
        let hash = try #require(PromptCacheBlockIndex.prefixBlockHashes(for: tokenIds, blockSize: 4).first)
        return (tokenIds, hash)
    }

    private static func store(
        _ payload: String,
        fixture: (tokenIds: [Int], hash: String),
        signature: PromptCacheSignature,
        root: URL,
        kind: MLXPersistentPromptCacheBlockPayloadKind
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: fixture.hash,
            blockSize: 4,
            tokenCount: 4,
            signature: signature,
            payload: Data(payload.utf8),
            payloadKind: kind,
            rootURL: root,
            now: date(1)
        )
    }

    private static func prefixData(
        tokens: [Int],
        signature: PromptCacheSignature,
        root: URL
    ) throws -> Data {
        let hit = try #require(try MLXPersistentPromptCacheBlockStore.lookupPrefix(
            tokenIds: tokens,
            signature: signature,
            blockSize: 4,
            rootURL: root
        ))
        return try Data(contentsOf: try #require(hit.dataURLs.first))
    }

    private static func snapshotData(
        tokens: [Int],
        signature: PromptCacheSignature,
        root: URL
    ) throws -> Data {
        let hit = try #require(try MLXPersistentPromptCacheBlockStore.lookupBestPrefixSnapshot(
            tokenIds: tokens,
            signature: signature,
            blockSize: 4,
            rootURL: root
        ))
        return try Data(contentsOf: hit.dataURL)
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func signature(
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize
    ) -> PromptCacheSignature {
        PromptCacheSignature(parameters: GenerateParameters(prefillStepSize: prefillStepSize))
    }

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
    }
}
