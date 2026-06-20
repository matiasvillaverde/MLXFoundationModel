import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX rotating KV quantization")
struct MLXRotatingKVQuantizationTests {
    @Test("dynamic quantization converts rotating caches and records diagnostics")
    func dynamicQuantizationConvertsRotatingCachesAndRecordsDiagnostics() async throws {
        var cache: [KVCache] = [
            Self.rotatingCache(offset: 12, maxSize: 64, keep: 8)
        ]

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: 4,
                kvGroupSize: 32,
                quantizedKVStart: 4
            )
        }
        let quantized = try #require(cache[0] as? QuantizedRotatingKVCache)
        let snapshot = try #require(Self.conversionSnapshots(from: recorded.events).last)

        #expect(quantized.maxSize == 64)
        #expect(quantized.metaState == ["8", "64", "256", "12", "12", "32", "4", "affine"])
        #expect(snapshot.offset == 12)
        #expect(snapshot.kvBits == 4)
        #expect(snapshot.kvGroupSize == 32)
        #expect(snapshot.quantizedKVStart == 4)
        #expect(!snapshot.quantizedKVSkipLastLayer)
        #expect(snapshot.convertedCount == 1)
    }

    @Test("dynamic quantization can keep the final layer cache in full precision")
    func dynamicQuantizationCanKeepFinalLayerCacheInFullPrecision() async throws {
        var cache: [KVCache] = [
            Self.rotatingCache(offset: 12, maxSize: 64, keep: 8),
            Self.rotatingCache(offset: 12, maxSize: 64, keep: 8)
        ]

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: 3,
                kvGroupSize: 32,
                quantizedKVStart: 4,
                skipLastLayer: true
            )
        }
        let snapshot = try #require(Self.conversionSnapshots(from: recorded.events).last)

        #expect(cache[0] is QuantizedRotatingKVCache)
        #expect(cache[1] is RotatingKVCache)
        #expect(snapshot.kvBits == 3)
        #expect(snapshot.quantizedKVSkipLastLayer)
        #expect(snapshot.convertedCount == 1)
    }

    @Test("dynamic quantization traverses composite cache children")
    func dynamicQuantizationTraversesCompositeCacheChildren() async throws {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = 10
        let rotatingCache = Self.rotatingCache(offset: 10, maxSize: 128, keep: 0)
        let compositeCache = CacheList(simpleCache, rotatingCache)
        var cache: [KVCache] = [compositeCache]

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: 4,
                kvGroupSize: 64,
                quantizedKVStart: 0
            )
        }
        let convertedComposite = try #require(cache[0] as? CacheList)
        let snapshot = try #require(Self.conversionSnapshots(from: recorded.events).last)

        #expect(convertedComposite.layoutCaches[0] is QuantizedKVCache)
        #expect(convertedComposite.layoutCaches[1] is QuantizedRotatingKVCache)
        #expect(snapshot.offset == 10)
        #expect(!snapshot.quantizedKVSkipLastLayer)
        #expect(snapshot.convertedCount == 2)
    }

    @Test("prompt cache layout includes quantized rotating metadata")
    func promptCacheLayoutIncludesQuantizedRotatingMetadata() {
        let layout = PromptCachePlanner.cacheLayoutFingerprint(
            for: [
                QuantizedRotatingKVCache(
                    maxSize: 128,
                    keep: 16,
                    groupSize: 32,
                    bits: 4
                )
            ]
        )

        #expect(layout == [
            "main[0]:QuantizedRotatingKVCache(keep:16,maxSize:128,step:256," +
                "groupSize:32,bits:4,mode:affine)"
        ])
    }

    @Test(
        "quantized rotating cache keeps the sliding window bounded during decode",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func quantizedRotatingCacheKeepsSlidingWindowBoundedDuringDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let cache = QuantizedRotatingKVCache(
                maxSize: 4,
                keep: 1,
                groupSize: 32,
                bits: 4
            )

            _ = cache.updateQuantized(
                keys: Self.tensor(start: 0, tokenCount: 3),
                values: Self.tensor(start: 100, tokenCount: 3)
            )
            _ = cache.updateQuantized(
                keys: Self.tensor(start: 12, tokenCount: 1),
                values: Self.tensor(start: 112, tokenCount: 1)
            )
            _ = cache.updateQuantized(
                keys: Self.tensor(start: 16, tokenCount: 1),
                values: Self.tensor(start: 116, tokenCount: 1)
            )

            let state = try #require(cache.getQuantizedState())

            #expect(cache.offset == 5)
            #expect(cache.metaState[4] == "2")
            #expect(state.0.0.dim(-2) == 4)
            #expect(state.1.0.dim(-2) == 4)
        }
    }

    private static func rotatingCache(
        offset: Int,
        maxSize: Int,
        keep: Int
    ) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxSize, keep: keep)
        cache.metaState = [
            String(keep),
            String(maxSize),
            "256",
            String(offset),
            String(min(offset, maxSize))
        ]
        return cache
    }

    private static func tensor(start: Int, tokenCount: Int) -> MLXArray {
        let headDimension = 32
        let values = (start ..< start + tokenCount * headDimension).map(Float.init)
        return MLXArray(values).reshaped([1, 1, tokenCount, headDimension])
    }

    private static func conversionSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXQuantizedKVConversionSnapshot] {
        events.compactMap { event in
            guard case .quantizedKVConversion(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }
}
