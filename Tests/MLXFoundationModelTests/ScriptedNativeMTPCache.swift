import MLX
@testable import MLXLocalModels

final class ScriptedNativeMTPCache: BaseKVCache {
    private(set) var trimmedTokenCount = 0

    deinit {
        // Required by the strict test lint profile.
    }

    func advance(by tokenCount: Int) {
        offset += tokenCount
    }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        advance(by: keys.dim(2))
        return (keys, values)
    }

    override var isTrimmable: Bool {
        true
    }

    override func trim(_ tokenCount: Int) -> Int {
        let trimmed = min(offset, tokenCount)
        offset -= trimmed
        trimmedTokenCount += trimmed
        return trimmed
    }

    override func copy() -> KVCache {
        let cache = ScriptedNativeMTPCache()
        cache.offset = offset
        return cache
    }
}
