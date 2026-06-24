import MLX
@testable import MLXLocalModels
import Testing

@Suite("KV cache architecture")
struct KVCacheArchitectureTests {
    @Test("plans stepped cache appends")
    func plansSteppedCacheAppends() {
        let initial = KVCacheAppendPlan(
            offset: 0,
            incomingTokenCount: 3,
            currentCapacity: nil,
            step: 4
        )
        let growth = KVCacheAppendPlan(
            offset: 3,
            incomingTokenCount: 2,
            currentCapacity: 4,
            step: 4
        )
        let reuse = KVCacheAppendPlan(
            offset: 2,
            incomingTokenCount: 1,
            currentCapacity: 4,
            step: 4
        )

        #expect(initial.writeRange == 0 ..< 3)
        #expect(initial.retainedLength == 0)
        #expect(initial.additionalCapacity == 4)
        #expect(growth.writeRange == 3 ..< 5)
        #expect(growth.retainedLength == 3)
        #expect(growth.additionalCapacity == 4)
        #expect(reuse.writeRange == 2 ..< 3)
        #expect(reuse.additionalCapacity == 0)
    }

    @Test("simple cache overwrites after trim without reallocating")
    func simpleCacheOverwritesAfterTrimWithoutReallocating() {
        Device.withDefaultDevice(.cpu) {
            let cache = KVCacheSimple()
            cache.step = 4

            _ = cache.update(
                keys: Self.tensor(start: 0, tokenCount: 3),
                values: Self.tensor(start: 100, tokenCount: 3)
            )
            #expect(cache.offset == 3)
            #expect(cache.state[0].shape == [1, 1, 3, 2])

            #expect(cache.trim(1) == 1)
            let updated = cache.update(
                keys: Self.tensor(start: 50, tokenCount: 1),
                values: Self.tensor(start: 150, tokenCount: 1)
            )

            #expect(cache.offset == 3)
            #expect(updated.0.shape == [1, 1, 3, 2])
            #expect(updated.0.asArray(Float.self) == [0, 1, 2, 3, 50, 51])
            #expect(updated.1.asArray(Float.self) == [100, 101, 102, 103, 150, 151])
        }
    }

    @Test("chunked cache preserves local window after front trim")
    func chunkedCachePreservesLocalWindowAfterFrontTrim() {
        Device.withDefaultDevice(.cpu) {
            let cache = ChunkedKVCache(chunkSize: 2)
            cache.step = 4

            _ = cache.update(
                keys: Self.tensor(start: 0, tokenCount: 3),
                values: Self.tensor(start: 100, tokenCount: 3)
            )
            cache.maybeTrimFront()
            let updated = cache.update(
                keys: Self.tensor(start: 50, tokenCount: 1),
                values: Self.tensor(start: 150, tokenCount: 1)
            )

            #expect(cache.offset == 4)
            #expect(cache.metaState == ["2", "1"])
            #expect(updated.0.shape == [1, 1, 3, 2])
            #expect(updated.0.asArray(Float.self) == [2, 3, 4, 5, 50, 51])
            #expect(updated.1.asArray(Float.self) == [102, 103, 104, 105, 150, 151])
        }
    }

    @Test("quantized cache grows with compact state metadata")
    func quantizedCacheGrowsWithCompactStateMetadata() {
        Device.withDefaultDevice(.cpu) {
            let cache = QuantizedKVCache(groupSize: 32, bits: 4)
            _ = cache.updateQuantized(
                keys: Self.tensor(start: 0, tokenCount: 2, headDimension: 32),
                values: Self.tensor(start: 100, tokenCount: 2, headDimension: 32)
            )
            _ = cache.updateQuantized(
                keys: Self.tensor(start: 1_000, tokenCount: 1, headDimension: 32),
                values: Self.tensor(start: 2_000, tokenCount: 1, headDimension: 32)
            )
            let state = cache.state

            #expect(cache.offset == 3)
            #expect(cache.metaState == ["256", "3", "32", "4"])
            #expect(state.count == 6)
            #expect(state[0].dim(-2) == 3)
            #expect(state[3].dim(-2) == 3)
        }
    }

    private static func tensor(
        start: Float,
        tokenCount: Int,
        headDimension: Int = 2
    ) -> MLXArray {
        MLXArray((0 ..< tokenCount * headDimension).map { start + Float($0) })
            .reshaped([1, 1, tokenCount, headDimension])
    }
}
