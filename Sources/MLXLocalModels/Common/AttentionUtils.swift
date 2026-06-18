import Foundation
import MLX
import MLXFast

internal struct AttentionWithCacheUpdateResult {
    internal let output: MLXArray
    internal let keys: MLXArray
    internal let values: MLXArray
}

/// Attention utilities that match Python mlx-lm's interface
///
/// This provides a single function that automatically routes to quantized or regular
/// attention based on cache type, matching Python's `scaled_dot_product_attention`

/// Automatic attention with cache update
///
/// This function matches Python's `scaled_dot_product_attention` in base.py:
/// - Detects if cache is `QuantizedKVCache` using `isinstance` pattern
/// - Routes to `quantizedScaledDotProductAttention` or `MLXFast.scaledDotProductAttention`
/// - Handles cache updating automatically
/// - Transparent to models - they just call this function
///
/// **Usage in models:**
/// ```swift
/// let output = attentionWithCacheUpdate(
///     queries: queries,
///     keys: keys,
///     values: values,
///     cache: cache,
///     scale: scale,
///     mask: mask
/// )
/// ```
///
/// - Parameters:
///   - queries: Query tensor [B, nHeads, L, D]
///   - keys: Raw key tensor to be cached [B, nKVHeads, L, D]
///   - values: Raw value tensor to be cached [B, nKVHeads, L, D]
///   - cache: Cache instance (any type)
///   - scale: Attention scale factor
///   - mask: Attention mask
/// - Returns: Attention output [B, nHeads, L, D]
internal func attentionWithCacheUpdate(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none
) -> MLXArray {
    attentionWithCacheUpdateReturningKV(
        queries: queries,
        keys: keys,
        values: values,
        cache: cache,
        scale: scale,
        mask: mask
    ).output
}

internal func attentionWithCacheUpdateReturningKV(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    maskForKeySequenceLength: ((Int) -> MLXFast.ScaledDotProductAttentionMaskMode)? = nil
) -> AttentionWithCacheUpdateResult {
    guard let cache else {
        let effectiveMask = maskForKeySequenceLength?(keys.dim(-2)) ?? mask
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: effectiveMask
        )
        return AttentionWithCacheUpdateResult(output: output, keys: keys, values: values)
    }

    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys, values: values)
        let materialized = materializeQuantizedKV(
            keys: quantizedKeys,
            values: quantizedValues,
            cache: quantizedKVCache
        )
        let effectiveMask = maskForKeySequenceLength?(materialized.keys.dim(-2)) ?? mask
        let output = quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: effectiveMask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode
        )
        return AttentionWithCacheUpdateResult(
            output: output,
            keys: materialized.keys,
            values: materialized.values
        )
    }

    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    let effectiveMask = maskForKeySequenceLength?(cachedKeys.dim(-2)) ?? mask
    let output = MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: cachedKeys,
        values: cachedValues,
        scale: scale,
        mask: effectiveMask
    )
    return AttentionWithCacheUpdateResult(
        output: output,
        keys: cachedKeys,
        values: cachedValues
    )
}

internal func materializedKVState(
    from cache: KVCache
) -> (keys: MLXArray, values: MLXArray)? {
    if let quantizedCache = cache as? QuantizedKVCacheProtocol,
        let quantizedState = quantizedCache.getQuantizedState() {
        return materializeQuantizedKV(
            keys: quantizedState.0,
            values: quantizedState.1,
            cache: quantizedCache
        )
    }

    let state = cache.state
    guard state.count >= 2 else { return nil }
    return (keys: state[0], values: state[1])
}

internal func updateCacheReturningMaterializedKV(
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache
) -> (keys: MLXArray, values: MLXArray) {
    if let quantizedCache = cache as? QuantizedKVCacheProtocol {
        _ = quantizedCache.updateQuantized(keys: keys, values: values)
        guard let materialized = materializedKVState(from: quantizedCache) else {
            fatalError("Quantized KV cache update produced no materialized state")
        }
        return materialized
    }

    let updated = cache.update(keys: keys, values: values)
    return (keys: updated.0, values: updated.1)
}

internal func expectedKVCacheKeySequenceLength(
    cache: KVCache?,
    newTokenCount: Int
) -> Int {
    guard let cache else { return newTokenCount }
    let updatedCount = cache.offset + newTokenCount
    guard let maxSize = cache.maxSize else { return updatedCount }
    return min(maxSize, updatedCount)
}

private func materializeQuantizedKV(
    keys: (MLXArray, MLXArray, MLXArray?),
    values: (MLXArray, MLXArray, MLXArray?),
    cache: QuantizedKVCacheProtocol
) -> (keys: MLXArray, values: MLXArray) {
    let materializedKeys = dequantized(
        keys.0,
        scales: keys.1,
        biases: keys.2,
        groupSize: cache.groupSize,
        bits: cache.bits,
        mode: cache.mode
    )
    let materializedValues = dequantized(
        values.0,
        scales: values.1,
        biases: values.2,
        groupSize: cache.groupSize,
        bits: cache.bits,
        mode: cache.mode
    )
    return (keys: materializedKeys, values: materializedValues)
}
