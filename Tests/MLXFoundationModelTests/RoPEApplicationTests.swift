import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("RoPE application")
struct RoPEApplicationTests {
    @Test("uses zero scalar offset when no cache is available")
    func usesZeroOffsetWithoutCache() {
        let rope = RecordingRoPE()

        _ = applyRotaryPosition(rope, to: MLXArray([Float(1)]), cache: nil)

        #expect(rope.scalarOffsets == [0])
        #expect(rope.batchOffsets.isEmpty)
    }

    @Test("uses scalar cache offset for normal KV caches")
    func usesScalarCacheOffset() {
        let rope = RecordingRoPE()
        let cache = OffsetCache(offset: 7)

        _ = applyRotaryPosition(rope, to: MLXArray([Float(1)]), cache: cache)

        #expect(rope.scalarOffsets == [7])
        #expect(rope.batchOffsets.isEmpty)
    }

    @Test("uses batch offsets when the cache provides per-row positions")
    func usesBatchCacheOffsets() {
        let rope = RecordingRoPE()
        let cache = BatchOffsetCache(offset: 11, batchOffset: [3, 9])

        _ = applyRotaryPosition(rope, to: MLXArray([Float(1)]), cache: cache)

        #expect(rope.scalarOffsets.isEmpty)
        #expect(rope.batchOffsets == [[3, 9]])
    }

    private final class RecordingRoPE: Module, OffsetLayer, ArrayOffsetLayer {
        private(set) var scalarOffsets: [Int] = []
        private(set) var batchOffsets: [[Int32]] = []

        deinit {
            // Required by the strict test lint profile.
        }

        func callAsFunction(_ input: MLXArray, offset: Int) -> MLXArray {
            scalarOffsets.append(offset)
            return input
        }

        func callAsFunction(_ input: MLXArray, offset: MLXArray) -> MLXArray {
            batchOffsets.append(offset.asArray(Int32.self))
            return input
        }
    }

    private class OffsetCache: BaseKVCache {
        init(offset: Int) {
            super.init()
            self.offset = offset
        }

        deinit {
            // Required by the strict test lint profile.
        }
    }

    private final class BatchOffsetCache: OffsetCache, BatchPositionedKVCache {
        let batchOffset: MLXArray

        init(offset: Int, batchOffset: [Int32]) {
            self.batchOffset = MLXArray(batchOffset)
            super.init(offset: offset)
        }

        deinit {
            // Required by the strict test lint profile.
        }
    }
}
