import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache tip compaction", .serialized)
struct MLXTipCompactionTests {
    @Test("compacts superseded rotating tips without touching the previous fallback")
    func compactsSupersededRotatingTips() async throws {
        try await MLXTipCompactionSupport.withHotCache(limitBytes: 10_000) {
            let fixture = try MLXTipCompactionSupport.storeThreeBlocks(
                signature: MLXTipCompactionSupport.rotatingSignature()
            )
            let compactPayload = Data([9, 8, 7])

            let first = try MLXTipCompactionSupport.recordExtension(
                fixture.records[0],
                fixture.records[1],
                root: fixture.root,
                payload: compactPayload
            )
            let second = try MLXTipCompactionSupport.recordExtension(
                fixture.records[1],
                fixture.records[2],
                root: fixture.root,
                payload: compactPayload
            )

            try MLXTipCompactionSupport.expectCompacted(
                first,
                second,
                fixture: fixture,
                payload: compactPayload
            )
        }
    }

    @Test("does not compact pure KV cache signatures")
    func doesNotCompactPureKVSignatures() async throws {
        try await MLXTipCompactionSupport.withHotCache(limitBytes: 10_000) {
            let fixture = try MLXTipCompactionSupport.storeThreeBlocks(
                signature: MLXTipCompactionSupport.kvSignature()
            )
            let payload = Data([1])

            let first = try MLXTipCompactionSupport.recordExtension(
                fixture.records[0],
                fixture.records[1],
                root: fixture.root,
                payload: payload
            )
            let second = try MLXTipCompactionSupport.recordExtension(
                fixture.records[1],
                fixture.records[2],
                root: fixture.root,
                payload: payload
            )

            #expect(first == nil)
            #expect(second == nil)
            try MLXTipCompactionSupport.expectBlockPayload(
                fixture.records[0],
                root: fixture.root,
                equals: Data(repeating: 1, count: 50)
            )
        }
    }

    @Test("segment store keeps older rotating tips compacted across longer prompts")
    func segmentStoreKeepsOlderTipsCompacted() async throws {
        try await MLXTipCompactionSupport.withHotCache(limitBytes: 10_000) {
            let root = try MLXTipCompactionSupport.makeTemporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let signature = MLXTipCompactionSupport.rotatingSignature()

            try MLXTipCompactionSupport.storeSegments(
                tokens: Array(0 ..< 8),
                signature: signature,
                root: root
            )
            try MLXTipCompactionSupport.storeSegments(
                tokens: Array(0 ..< 12),
                signature: signature,
                root: root
            )
            try MLXTipCompactionSupport.storeSegments(
                tokens: Array(0 ..< 16),
                signature: signature,
                root: root
            )

            let hashes = PromptCacheBlockIndex.prefixBlockHashes(for: Array(0 ..< 16), blockSize: 4)
            try MLXTipCompactionSupport.expectSegmentKinds(hashes: hashes, signature: signature, root: root)
        }
    }
}
