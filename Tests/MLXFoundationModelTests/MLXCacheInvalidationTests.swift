import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache signature invalidation")
struct MLXCacheInvalidationTests {
    @Test("removes stale records in the same cache scope")
    func removesStaleRecordsInTheSameCacheScope() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let hash = Self.hash("4")
        let expected = Self.signature(identity: "shared", cacheLayout: ["KVCache"])
        let stale = Self.signature(identity: "shared", cacheLayout: ["RotatingKVCache"])
        let otherPrompt = Self.signature(identity: "other", cacheLayout: ["RotatingKVCache"])
        let fixture = try Self.storeInvalidationFixture(
            hash: hash,
            staleSignature: stale,
            expectedSignature: expected,
            otherSignature: otherPrompt,
            root: root
        )

        let removed = try MLXPersistentPromptCacheBlockStore.invalidateStaleSignatures(
            expectedSignature: expected,
            rootURL: root
        )
        let remaining = try MLXPersistentPromptCacheBlockStore.scan(rootURL: root)

        #expect(removed == [fixture.stale])
        #expect(remaining.map(\.storageHash).sorted() == [
            fixture.expected.storageHash,
            fixture.other.storageHash
        ].sorted())
    }

    @Test("is disabled without a cache identity")
    func isDisabledWithoutACacheIdentity() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let hash = Self.hash("6")
        let expected = Self.signature(cacheLayout: ["KVCache"])
        let stale = Self.signature(kvBits: 4, cacheLayout: ["QuantizedKVCache"])

        let staleRecord = try Self.store(
            hash: hash,
            bytes: 10,
            access: 1,
            root: root,
            signature: stale
        )
        let expectedRecord = try Self.store(
            hash: hash,
            bytes: 20,
            access: 2,
            root: root,
            signature: expected
        )

        let removed = try MLXPersistentPromptCacheBlockStore.invalidateStaleSignatures(
            expectedSignature: expected,
            rootURL: root
        )
        let remaining = try MLXPersistentPromptCacheBlockStore.scan(rootURL: root)

        #expect(removed.isEmpty)
        #expect(remaining.map(\.storageHash).sorted() == [
            expectedRecord.storageHash,
            staleRecord.storageHash
        ].sorted())
    }

    @Test("records diagnostics")
    func recordsDiagnostics() async throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let expected = Self.signature(identity: "shared", cacheLayout: ["KVCache"])
        let stale = Self.signature(identity: "shared", cacheLayout: ["RotatingKVCache"])
        _ = try Self.store(
            hash: Self.hash("7"),
            bytes: 10,
            access: 1,
            root: root,
            signature: stale
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            try MLXPersistentPromptCacheBlockStore.invalidateStaleSignatures(
                expectedSignature: expected,
                rootURL: root,
                payloadKinds: [.generic]
            )
        }
        let snapshot = try #require(Self.invalidationSnapshots(from: recorded.events).last)

        #expect(snapshot.stage == .staleSignatureSweep)
        #expect(snapshot.candidateCount == 1)
        #expect(snapshot.removedCount == 1)
        #expect(snapshot.payloadKinds == ["generic"])
    }

    private static func store(
        hash: String,
        bytes: Int,
        access: TimeInterval,
        root: URL,
        signature: PromptCacheSignature
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try MLXPersistentPromptCacheBlockStore.storeBlock(
            blockHash: hash,
            tokenCount: bytes,
            signature: signature,
            payload: Data(repeating: 0, count: bytes),
            rootURL: root,
            now: date(access)
        )
    }

    private static func storeInvalidationFixture(
        hash: String,
        staleSignature: PromptCacheSignature,
        expectedSignature: PromptCacheSignature,
        otherSignature: PromptCacheSignature,
        root: URL
    ) throws -> InvalidationFixture {
        let stale = try store(
            hash: hash,
            bytes: 10,
            access: 1,
            root: root,
            signature: staleSignature
        )
        let expected = try store(
            hash: hash,
            bytes: 20,
            access: 2,
            root: root,
            signature: expectedSignature
        )
        let other = try store(
            hash: Self.hash("5"),
            bytes: 30,
            access: 3,
            root: root,
            signature: otherSignature
        )
        return InvalidationFixture(stale: stale, expected: expected, other: other)
    }

    private static func invalidationSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXPersistentCacheInvalidationSnapshot] {
        events.compactMap { event in
            guard case .persistentCacheInvalidation(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private struct InvalidationFixture {
        let stale: MLXPersistentPromptCacheBlockRecord
        let expected: MLXPersistentPromptCacheBlockRecord
        let other: MLXPersistentPromptCacheBlockRecord
    }

    private static func signature(
        kvBits: Int? = nil,
        identity: String? = nil,
        cacheLayout: [String] = []
    ) -> PromptCacheSignature {
        PromptCacheSignature(
            parameters: GenerateParameters(kvBits: kvBits),
            cacheLayout: cacheLayout.isEmpty ? nil : cacheLayout,
            promptCacheIdentity: identity.map(PromptCacheIdentity.init(stableFingerprint:))
        )
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root
    }

    private static func hash(_ character: Character) -> String {
        String(repeating: String(character), count: 64)
    }

    private static func date(_ offset: TimeInterval) -> Date {
        Date(timeIntervalSinceReferenceDate: offset)
    }
}
