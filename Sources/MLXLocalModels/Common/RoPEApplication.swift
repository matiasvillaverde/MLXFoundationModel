import MLX
import MLXNN

/// Cache type that can provide one RoPE offset per sequence in a batch.
internal protocol BatchPositionedKVCache: KVCache {
    var batchOffset: MLXArray { get }
}

/// Applies RoPE with the offset form required by the current cache.
internal func applyRotaryPosition<R: RoPELayer>(_ rope: R, to input: MLXArray, cache: KVCache?)
    -> MLXArray
{
    switch ropeOffset(from: cache) {
    case .scalar(let offset):
        return rope(input, offset: offset)
    case .batch(let offset):
        return rope(input, offset: offset)
    }
}

private enum RoPEPositionOffset {
    case scalar(Int)
    case batch(MLXArray)
}

@inline(__always)
private func ropeOffset(from cache: KVCache?) -> RoPEPositionOffset {
    guard let cache else {
        return .scalar(0)
    }

    if let batchCache = cache as? BatchPositionedKVCache {
        return .batch(batchCache.batchOffset)
    }

    return .scalar(cache.offset)
}
