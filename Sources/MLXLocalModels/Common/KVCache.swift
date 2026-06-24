import Foundation
import MLX
import MLXNN

/// Interface for attention key/value caches used during autoregressive decoding.
///
/// See ``LanguageModel/newCache(parameters:)``
internal protocol KVCache: Evaluatable {
    /// get the current offset
    var offset: Int { get }

    /// get the maximum size (if any)
    var maxSize: Int? { get }

    /// update the cache with new keys and values and return all keys/values
    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    /// make an attention mask for this cache's current offset/window behavior
    func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    /// make an independent cache copy
    func copy() -> KVCache

    /// get the current state for serialization
    var state: [MLXArray] { get set }

    /// get/set metadata state as string array for serialization
    var metaState: [String] { get set }

    /// whether this cache can be trimmed
    var isTrimmable: Bool { get }

    /// trim n tokens from the cache, returning actual number trimmed
    @discardableResult
    func trim(_ n: Int) -> Int
}

/// Protocol for caches that support efficient quantized operations
///
/// **Usage Example:**
/// ```swift
/// // Efficient quantized path
/// if let quantizedCache = cache as? QuantizedKVCacheProtocol {
///     let (qKeys, qValues) = quantizedCache.updateQuantized(keys: k, values: v)
///     // Use native quantized operations
///     let scores = quantizedMM(queries, w: qKeys.0, scales: qKeys.1, biases: qKeys.2, ...)
/// } else {
///     // Regular path
///     let (k, v) = cache.update(keys: k, values: v)
///     let output = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, ...)
/// }
/// ```
internal protocol QuantizedKVCacheProtocol: KVCache {
    /// The quantization group size used
    var groupSize: Int { get }

    /// The number of quantization bits used
    var bits: Int { get }

    /// The quantization mode used
    var mode: QuantizationMode { get }

    /// Update cache and return quantized tuples for maximum efficiency
    ///
    /// - Parameters:
    ///   - keys: New key data to add to cache
    ///   - values: New value data to add to cache
    /// - Returns: Quantized tuples (keys, values) as ((weight, scales, biases), (weight, scales, biases))
    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    /// Get current quantized state without updating
    ///
    /// Useful for accessing cached data without adding new tokens.
    /// - Returns: Current quantized state, or nil if cache is empty
    func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )?
}

internal typealias QuantizedKVStorage = (MLXArray, MLXArray, MLXArray?)

private struct QuantizedKVState {
    var weights: MLXArray
    var scales: MLXArray
    var biases: MLXArray?

    init(_ storage: QuantizedKVStorage) {
        self.weights = storage.0
        self.scales = storage.1
        self.biases = storage.2
    }

    init(weights: MLXArray, scales: MLXArray, biases: MLXArray?) {
        self.weights = weights
        self.scales = scales
        self.biases = biases
    }

    var storage: QuantizedKVStorage {
        (weights, scales, biases)
    }

    var arrays: [MLXArray] {
        [weights, scales, biases].compactMap { $0 }
    }

    var tokenCapacity: Int {
        weights.dim(-2)
    }

    func map(_ transform: (MLXArray) -> MLXArray) -> QuantizedKVState {
        QuantizedKVState(
            weights: transform(weights),
            scales: transform(scales),
            biases: biases.map(transform)
        )
    }

    func prefix(upTo end: Int) -> QuantizedKVState {
        map { $0[.ellipsis, ..<end, 0...] }
    }

    func range(_ range: Range<Int>) -> QuantizedKVState {
        map { $0[.ellipsis, range, 0...] }
    }

    mutating func write(_ source: QuantizedKVState, range: Range<Int>) {
        weights[.ellipsis, range, 0...] = source.weights
        scales[.ellipsis, range, 0...] = source.scales
        if source.biases != nil {
            guard let sourceBiases = source.biases, let targetBiases = biases else {
                fatalError("Quantized KV bias layout mismatch")
            }
            targetBiases[.ellipsis, range, 0...] = sourceBiases
        }
    }

    static func quantizing(
        _ array: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> QuantizedKVState {
        let quantizedArray = quantized(array, groupSize: groupSize, bits: bits, mode: mode)
        return QuantizedKVState(
            weights: quantizedArray.wq,
            scales: quantizedArray.scales,
            biases: quantizedArray.biases
        )
    }

    static func zeros(
        shape: [Int],
        headDimension: Int,
        dtype: DType,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode
    ) -> QuantizedKVState {
        quantizing(
            MLXArray.zeros(shape + [headDimension], dtype: dtype),
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    static func concatenated(_ states: [QuantizedKVState]) -> QuantizedKVState {
        guard let first = states.first else {
            fatalError("Cannot concatenate an empty quantized KV state")
        }
        guard states.count > 1 else { return first }

        return QuantizedKVState(
            weights: MLX.concatenated(states.map(\.weights), axis: -2),
            scales: MLX.concatenated(states.map(\.scales), axis: -2),
            biases: first.biases == nil
                ? nil
                : MLX.concatenated(states.compactMap(\.biases), axis: -2)
        )
    }
}

internal struct KVCacheAppendPlan: Equatable, Sendable {
    internal let writeRange: Range<Int>
    internal let retainedLength: Int
    internal let additionalCapacity: Int

    internal init(
        offset: Int,
        baseOffset: Int = 0,
        incomingTokenCount: Int,
        currentCapacity: Int?,
        step: Int
    ) {
        precondition(step > 0, "cache growth step must be positive")
        precondition(incomingTokenCount >= 0, "incomingTokenCount cannot be negative")

        let localOffset = max(0, offset - baseOffset)
        let requiredCapacity = localOffset + incomingTokenCount
        self.writeRange = localOffset ..< requiredCapacity
        self.retainedLength = min(localOffset, currentCapacity ?? 0)

        guard let currentCapacity, requiredCapacity <= currentCapacity else {
            let existingCapacity = currentCapacity ?? 0
            let targetCapacity = Self.roundUp(requiredCapacity, step: step)
            self.additionalCapacity = max(0, targetCapacity - existingCapacity)
            return
        }

        self.additionalCapacity = 0
    }

    internal var needsGrowth: Bool {
        additionalCapacity > 0
    }

    private static func roundUp(_ value: Int, step: Int) -> Int {
        guard value > 0 else { return step }
        return ((value + step - 1) / step) * step
    }
}

private struct DenseKVStorage {
    var keys: MLXArray
    var values: MLXArray

    var capacity: Int {
        keys.dim(2)
    }

    func state(offset: Int) -> [MLXArray] {
        guard offset < capacity else {
            return [keys, values]
        }
        return [
            keys[.ellipsis, ..<offset, 0...],
            values[.ellipsis, ..<offset, 0...]
        ]
    }

    func prefix(_ length: Int) -> DenseKVStorage {
        DenseKVStorage(
            keys: keys[.ellipsis, ..<length, 0...],
            values: values[.ellipsis, ..<length, 0...]
        )
    }

    mutating func write(keys newKeys: MLXArray, values newValues: MLXArray, range: Range<Int>) {
        keys[.ellipsis, range, 0...] = newKeys
        values[.ellipsis, range, 0...] = newValues
    }
}

private func makeDenseKVStorage(
    batchSize: Int,
    headCount: Int,
    capacity: Int,
    keyHeadDimension: Int,
    valueHeadDimension: Int,
    keyDType: DType,
    valueDType: DType
) -> DenseKVStorage {
    DenseKVStorage(
        keys: MLXArray.zeros(
            [batchSize, headCount, capacity, keyHeadDimension],
            dtype: keyDType
        ),
        values: MLXArray.zeros(
            [batchSize, headCount, capacity, valueHeadDimension],
            dtype: valueDType
        )
    )
}

private func appendDenseCapacity(
    to storage: DenseKVStorage?,
    plan: KVCacheAppendPlan,
    keys: MLXArray,
    values: MLXArray
) -> DenseKVStorage {
    let allocation = makeDenseKVStorage(
        batchSize: keys.dim(0),
        headCount: keys.dim(1),
        capacity: plan.additionalCapacity,
        keyHeadDimension: keys.dim(3),
        valueHeadDimension: values.dim(3),
        keyDType: keys.dtype,
        valueDType: values.dtype
    )

    guard let storage else {
        return allocation
    }

    let retained = storage.prefix(plan.retainedLength)
    return DenseKVStorage(
        keys: concatenated([retained.keys, allocation.keys], axis: 2),
        values: concatenated([retained.values, allocation.values], axis: 2)
    )
}

/// Base cache implementation providing default behaviors.
open class BaseKVCache: KVCache {
    internal var offset: Int = 0
    internal var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else { return .none }

        if returnArray || windowSize.map({ offset + n > $0 }) == true {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }
        return .causal
    }

    func copy() -> KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get {
            []
        }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }
}

func createCausalMask(n: Int, offset: Int, windowSize: Int? = nil) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    return mask
}

/// Create an attention mask using the parameters from the KVCache.
///
/// See also ``MultiHeadAttention/createAdditiveCausalMask(_:dtype:)`` -- same idea
/// but doesn't honor the cache offset.
@_disfavoredOverload
internal func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

internal func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if let firstCache = cache?.first {
        return firstCache.makeMask(
            n: t,
            windowSize: firstCache.maxSize,
            returnArray: returnArray
        )
    }

    return BaseKVCache().makeMask(n: t, windowSize: nil, returnArray: returnArray)
}

internal func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }
    if returnArray || windowSize.map({ n > $0 }) == true {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

internal func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    makeAttentionMask(
        n: h.dim(1),
        cache: cache,
        windowSize: windowSize,
        returnArray: returnArray
    )
}

internal func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    cache?.makeMask(N: h.dim(1))
}

/// Growable full-context KV cache used by normal attention layers.
internal class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    private var storage: DenseKVStorage?
    internal var step = 256

    internal var keys: MLXArray? {
        get { storage?.keys }
        set {
            storage = Self.storage(keys: newValue, values: values)
        }
    }

    internal var values: MLXArray? {
        get { storage?.values }
        set {
            storage = Self.storage(keys: keys, values: newValue)
        }
    }

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [keys, values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        var storage = self.storage
        let plan = KVCacheAppendPlan(
            offset: offset,
            incomingTokenCount: keys.dim(2),
            currentCapacity: storage?.capacity,
            step: step
        )

        if plan.needsGrowth {
            storage = appendDenseCapacity(to: storage, plan: plan, keys: keys, values: values)
        }

        guard var storage else {
            fatalError("KV cache storage was not initialized")
        }

        storage.write(keys: keys, values: values, range: plan.writeRange)
        self.storage = storage
        offset = plan.writeRange.upperBound

        return (
            storage.keys[.ellipsis, ..<offset, 0...],
            storage.values[.ellipsis, ..<offset, 0...]
        )
    }

    public override var state: [MLXArray] {
        get {
            storage?.state(offset: offset) ?? []
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            storage = DenseKVStorage(keys: newValue[0], values: newValue[1])
            offset = newValue[0].dim(2)
        }
    }

    public override var metaState: [String] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("KVCacheSimple should not have metaState.")
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    override func copy() -> KVCache {
        let new = KVCacheSimple()
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        return new
    }

    /// Convert to quantized cache for maximum efficiency
    ///
    /// Use `updateQuantized()` and `quantizedScaledDotProductAttention()` for zero-overhead operation.
    internal func toQuantized(
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) -> QuantizedKVCache {
        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        quantizedCache.offset = self.offset

        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]

            let quantizedKeys = quantized(
                currentKeys,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
            let quantizedValues = quantized(
                currentValues,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )

            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }
        }

        return quantizedCache
    }

    internal var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }

    fileprivate static func storage(keys: MLXArray?, values: MLXArray?) -> DenseKVStorage? {
        guard let keys, let values else {
            return nil
        }
        return DenseKVStorage(keys: keys, values: values)
    }

    fileprivate func replaceStorage(_ storage: DenseKVStorage?) {
        self.storage = storage
    }
}

/// Rotating KV cache for sliding window attention
internal class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    private var keep: Int
    private var keys: MLXArray?
    private var values: MLXArray?
    private var maxCacheSize: Int
    private var step: Int
    private var idx: Int = 0

    public override var maxSize: Int? { maxCacheSize }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        // Rearrange the cache into temporal order, slicing off the end if unused
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            // Put the keys/values in temporal order to preserve context
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)
            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let S = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset

        // May not have hit the max size yet, so potentially keep growing the cache
        if self.keys == nil
            || (prev >= self.keys!.dim(2) && self.keys!.dim(2) < maxCacheSize)
        {
            let newSize = min(step, maxCacheSize - prev)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            idx = prev
        }

        // Trim if needed
        let trimSize = self.keys!.dim(2) - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        // Rotate if we've hit the end
        if idx == maxCacheSize {
            idx = keep
        }

        // Assign
        self.keys![.ellipsis, idx ..< (idx + S), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + S), 0...] = values
        offset += S
        idx += S

        // Return the appropriate cache slice
        if offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let result =
            if keys.dim(2) == 1 {
                updateInPlace(keys: keys, values: values)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            // Note: RotatingKVCache doesn't set offset from keys like KVCache does
            // The offset is managed through meta_state
        }
    }

    public override var metaState: [String] {
        get {
            return [String(keep), String(maxCacheSize), String(step), String(offset), String(idx)]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("RotatingKVCache metaState must have exactly 5 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
        }
    }

    public override var isTrimmable: Bool {
        return offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else {
            return makeSingleTokenMask(windowSize: windowSize)
        }

        let actualWindowSize = windowSize ?? maxCacheSize
        let cappedOffset = min(maxCacheSize - 1, offset)
        if cappedOffset + n > actualWindowSize || returnArray {
            return .array(
                createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize)
            )
        }
        return .causal
    }

    override func copy() -> KVCache {
        let new = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    private func makeSingleTokenMask(
        windowSize: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let windowSize else { return .none }

        if offset >= windowSize, maxCacheSize > windowSize {
            let currentIdx = idx >= maxCacheSize ? 0 : idx
            let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            return .array(roll(mask, shift: currentIdx + 1))
        }
        return .none
    }

    internal var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx)"
    }

    /// Convert to quantized cache
    /// Note: This is complex due to the rotating nature and temporal ordering
    internal func toQuantized(
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) -> QuantizedRotatingKVCache {
        let quantizedCache = QuantizedRotatingKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        quantizedCache.metaState = metaState + [String(groupSize), String(bits), mode.rawValue]

        if let keys, let values {
            let orderedKeys = temporalOrder(keys)
            let orderedValues = temporalOrder(values)
            quantizedCache.loadQuantizedTemporalState(
                keys: orderedKeys,
                values: orderedValues
            )
        }

        return quantizedCache
    }
}

/// Quantized rotating KV cache for sliding-window attention.
///
/// The storage keeps the same physical rotation semantics as `RotatingKVCache`
/// during single-token decoding while exposing `QuantizedKVCacheProtocol` so
/// attention can stay on the quantized matmul path.
internal class QuantizedRotatingKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keep: Int
    private var keys: QuantizedKVStorage?
    private var values: QuantizedKVStorage?
    private var maxCacheSize: Int
    private var step: Int
    private var idx = 0

    internal let groupSize: Int
    internal let bits: Int
    internal let mode: QuantizationMode

    public override var maxSize: Int? { maxCacheSize }

    public init(
        maxSize: Int,
        keep: Int = 0,
        step: Int = 256,
        groupSize: Int = 64,
        bits: Int = 4,
        mode: QuantizationMode = .affine
    ) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    internal func loadQuantizedTemporalState(keys: MLXArray, values: MLXArray) {
        let qKeys = quantize(keys)
        let qValues = quantize(values)
        self.keys = retainedState(qKeys)
        self.values = retainedState(qValues)
        idx = self.keys?.0.dim(-2) ?? 0
    }

    internal func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        QuantizedKVStorage, QuantizedKVStorage
    ) {
        if keys.dim(2) == 1 {
            return updateInPlace(keys: keys, values: values)
        }
        return updateConcat(keys: keys, values: values)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedRotatingKVCache`. Use `updateQuantized` instead."
        )
    }

    internal func getQuantizedState() -> (QuantizedKVStorage, QuantizedKVStorage)? {
        guard let keys, let values else { return nil }
        return currentState(keys: keys, values: values)
    }

    public override var state: [MLXArray] {
        get {
            guard let state = getQuantizedState() else { return [] }
            return [
                state.0.0, state.0.1, state.0.2,
                state.1.0, state.1.1, state.1.2,
            ].compactMap { $0 }
        }
        set {
            switch newValue.count {
            case 4:
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedRotatingKVCache state must have exactly 6 or 4 arrays"
                )
            }
        }
    }

    public override var metaState: [String] {
        get {
            [
                String(keep),
                String(maxCacheSize),
                String(step),
                String(offset),
                String(idx),
                String(groupSize),
                String(bits),
                mode.rawValue,
            ]
        }
        set {
            guard newValue.count == 8 else {
                fatalError("QuantizedRotatingKVCache metaState must have exactly 8 values")
            }
            guard let keepVal = Int(newValue[0]),
                let maxSizeVal = Int(newValue[1]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4]),
                let groupSizeVal = Int(newValue[5]),
                let bitsVal = Int(newValue[6]),
                let modeVal = QuantizationMode(rawValue: newValue[7])
            else {
                fatalError("Failed to convert QuantizedRotatingKVCache metaState values")
            }
            guard groupSizeVal == groupSize, bitsVal == bits, modeVal == mode else {
                fatalError("QuantizedRotatingKVCache metaState does not match configuration")
            }
            keep = keepVal
            maxCacheSize = maxSizeVal
            step = stepVal
            offset = offsetVal
            idx = idxVal
        }
    }

    public override var isTrimmable: Bool {
        offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx = max(0, idx - trimmed)
        return trimmed
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard n > 1 else {
            return makeSingleTokenMask(windowSize: windowSize)
        }

        let actualWindowSize = windowSize ?? maxCacheSize
        let cappedOffset = min(maxCacheSize - 1, offset)
        if cappedOffset + n > actualWindowSize || returnArray {
            return .array(
                createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize)
            )
        }
        return .causal
    }

    override func copy() -> KVCache {
        let new = QuantizedRotatingKVCache(
            maxSize: maxCacheSize,
            keep: keep,
            step: step,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (
        QuantizedKVStorage, QuantizedKVStorage
    ) {
        let newKeys = quantize(keys)
        let newValues = quantize(values)

        if let currentKeys = self.keys, let currentValues = self.values {
            let orderedKeys = temporalOrder(currentKeys)
            let orderedValues = temporalOrder(currentValues)
            self.keys = retainedState(concatenatedState([orderedKeys, newKeys]))
            self.values = retainedState(concatenatedState([orderedValues, newValues]))
        } else {
            self.keys = retainedState(newKeys)
            self.values = retainedState(newValues)
        }

        offset += keys.dim(2)
        idx = self.keys?.0.dim(-2) ?? 0
        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized rotating cache not initialized")
        }
        return currentState(keys: currentKeys, values: currentValues)
    }

    private func updateInPlace(keys: MLXArray, values: MLXArray) -> (
        QuantizedKVStorage, QuantizedKVStorage
    ) {
        let batchSize = keys.dim(0)
        let kvHeadCount = keys.dim(1)
        let tokenCount = keys.dim(2)
        let keyHeadDimension = keys.dim(3)
        let valueHeadDimension = values.dim(3)
        let previousOffset = offset

        if self.keys == nil
            || (previousOffset >= self.keys!.0.dim(-2) && self.keys!.0.dim(-2) < maxCacheSize)
        {
            let newSize = min(step, maxCacheSize - previousOffset)
            let shape = [batchSize, kvHeadCount, newSize]
            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = expandedState(currentKeys, newShape: shape)
                self.values = expandedState(currentValues, newShape: shape)
            } else {
                self.keys = initialState(dim: keyHeadDimension, shape: shape, dtype: keys.dtype)
                self.values = initialState(dim: valueHeadDimension, shape: shape, dtype: values.dtype)
            }
            idx = previousOffset
        }

        guard var currentKeys = self.keys, var currentValues = self.values else {
            fatalError("Quantized rotating cache not initialized")
        }

        let trimSize = currentKeys.0.dim(-2) - maxCacheSize
        if trimSize > 0 {
            currentKeys = trimState(currentKeys, trimSize: trimSize)
            currentValues = trimState(currentValues, trimSize: trimSize)
            idx = maxCacheSize
        }

        if idx == maxCacheSize {
            idx = keep
        }

        let qKeys = quantize(keys)
        let qValues = quantize(values)
        write(qKeys, into: currentKeys, range: idx ..< (idx + tokenCount))
        write(qValues, into: currentValues, range: idx ..< (idx + tokenCount))

        self.keys = currentKeys
        self.values = currentValues
        offset += tokenCount
        idx += tokenCount

        return currentState(keys: currentKeys, values: currentValues)
    }

    private func initialState(
        dim: Int,
        shape: [Int],
        dtype: DType
    ) -> QuantizedKVStorage {
        let zeros = MLXArray.zeros(shape + [dim], dtype: dtype)
        return quantize(zeros)
    }

    private func expandedState(
        _ state: QuantizedKVStorage,
        newShape: [Int]
    ) -> QuantizedKVStorage {
        mapState(state) { array in
            let zeros = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
            return concatenated([array, zeros], axis: -2)
        }
    }

    private func retainedState(_ state: QuantizedKVStorage) -> QuantizedKVStorage {
        let tokenCount = state.0.dim(-2)
        guard tokenCount > maxCacheSize else { return state }

        let prefixCount = min(keep, maxCacheSize, tokenCount)
        let tailCapacity = max(0, maxCacheSize - prefixCount)
        var parts: [QuantizedKVStorage] = []

        appendRange(of: state, from: 0, to: prefixCount, into: &parts)
        if tailCapacity > 0 {
            let tailStart = max(prefixCount, tokenCount - tailCapacity)
            appendRange(of: state, from: tailStart, to: tokenCount, into: &parts)
        }

        return concatenatedState(parts)
    }

    private func trimState(
        _ state: QuantizedKVStorage,
        trimSize: Int
    ) -> QuantizedKVStorage {
        guard trimSize > 0 else { return state }

        let tokenCount = state.0.dim(-2)
        var parts: [QuantizedKVStorage] = []
        appendRange(of: state, from: 0, to: min(keep, tokenCount), into: &parts)
        appendRange(of: state, from: min(tokenCount, trimSize + keep), to: tokenCount, into: &parts)
        return concatenatedState(parts)
    }

    private func temporalOrder(_ state: QuantizedKVStorage) -> QuantizedKVStorage {
        let tokenCount = state.0.dim(-2)
        if idx == tokenCount {
            return state
        }
        if idx < offset {
            var parts: [QuantizedKVStorage] = []
            appendRange(of: state, from: 0, to: min(keep, tokenCount), into: &parts)
            appendRange(of: state, from: idx, to: tokenCount, into: &parts)
            appendRange(of: state, from: keep, to: min(idx, tokenCount), into: &parts)
            return concatenatedState(parts)
        }
        return slicePrefix(state, upTo: min(idx, tokenCount))
    }

    private func currentState(
        keys: QuantizedKVStorage,
        values: QuantizedKVStorage
    ) -> (QuantizedKVStorage, QuantizedKVStorage) {
        let tokenCount = keys.0.dim(-2)
        guard offset < tokenCount else {
            return (keys, values)
        }
        return (
            slicePrefix(keys, upTo: offset),
            slicePrefix(values, upTo: offset)
        )
    }

    private func makeSingleTokenMask(
        windowSize: Int?
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        guard let windowSize else { return .none }

        if offset >= windowSize, maxCacheSize > windowSize {
            let currentIdx = idx >= maxCacheSize ? 0 : idx
            let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
            let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)
            return .array(roll(mask, shift: currentIdx + 1))
        }
        return .none
    }

    private func quantize(_ array: MLXArray) -> QuantizedKVStorage {
        let quantizedArray = quantized(array, groupSize: groupSize, bits: bits, mode: mode)
        return (quantizedArray.wq, quantizedArray.scales, quantizedArray.biases)
    }

    private func mapState(
        _ state: QuantizedKVStorage,
        _ transform: (MLXArray) -> MLXArray
    ) -> QuantizedKVStorage {
        (transform(state.0), transform(state.1), state.2.map(transform))
    }

    private func slicePrefix(
        _ state: QuantizedKVStorage,
        upTo end: Int
    ) -> QuantizedKVStorage {
        mapState(state) { $0[.ellipsis, ..<end, 0...] }
    }

    private func sliceRange(
        _ state: QuantizedKVStorage,
        from start: Int,
        to end: Int
    ) -> QuantizedKVStorage {
        mapState(state) { $0[.ellipsis, start ..< end, 0...] }
    }

    private func appendRange(
        of state: QuantizedKVStorage,
        from start: Int,
        to end: Int,
        into parts: inout [QuantizedKVStorage]
    ) {
        guard start < end else { return }
        parts.append(sliceRange(state, from: start, to: end))
    }

    private func concatenatedState(_ states: [QuantizedKVStorage]) -> QuantizedKVStorage {
        guard let first = states.first else {
            fatalError("Cannot concatenate an empty quantized KV state")
        }
        guard states.count > 1 else { return first }

        let weights = concatenated(states.map { $0.0 }, axis: -2)
        let scales = concatenated(states.map { $0.1 }, axis: -2)
        let biases: MLXArray?
        if first.2 == nil {
            biases = nil
        } else {
            biases = concatenated(states.compactMap { $0.2 }, axis: -2)
        }
        return (weights, scales, biases)
    }

    private func write(
        _ source: QuantizedKVStorage,
        into target: QuantizedKVStorage,
        range: Range<Int>
    ) {
        target.0[.ellipsis, range, 0...] = source.0
        target.1[.ellipsis, range, 0...] = source.1
        if source.2 != nil {
            guard let sourceBiases = source.2, let targetBiases = target.2 else {
                fatalError("Quantized rotating cache bias layout mismatch")
            }
            targetBiases[.ellipsis, range, 0...] = sourceBiases
        }
    }
}

/// Quantized KV cache for memory efficiency using MLX quantization
internal class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: QuantizedKVState?
    private var values: QuantizedKVState?
    private let step: Int
    internal let groupSize: Int
    internal let bits: Int
    internal let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: keys.arrays)
        }
        if let values = values {
            arrays.append(contentsOf: values.arrays)
        }
        return arrays
    }

    /// Get current quantized keys and values as tuples (efficient access)
    /// - Returns: Tuple of ((keyWeight, keyScales, keyBiases), (valueWeight, valueScales, valueBiases))
    internal func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        return (
            keys.prefix(upTo: offset).storage,
            values.prefix(upTo: offset).storage
        )
    }

    internal func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        var currentKeys = self.keys
        var currentValues = self.values
        let plan = KVCacheAppendPlan(
            offset: offset,
            incomingTokenCount: keys.dim(2),
            currentCapacity: currentKeys?.tokenCapacity,
            step: step
        )

        if plan.needsGrowth {
            let shape = [keys.dim(0), keys.dim(1), plan.additionalCapacity]
            let emptyKeys = QuantizedKVState.zeros(
                shape: shape,
                headDimension: keys.dim(3),
                dtype: keys.dtype,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )
            let emptyValues = QuantizedKVState.zeros(
                shape: shape,
                headDimension: values.dim(3),
                dtype: values.dtype,
                groupSize: groupSize,
                bits: bits,
                mode: mode
            )

            if let existingKeys = currentKeys, let existingValues = currentValues {
                currentKeys = QuantizedKVState.concatenated([
                    existingKeys.prefix(upTo: plan.retainedLength),
                    emptyKeys
                ])
                currentValues = QuantizedKVState.concatenated([
                    existingValues.prefix(upTo: plan.retainedLength),
                    emptyValues
                ])
            } else {
                currentKeys = emptyKeys
                currentValues = emptyValues
            }
        }

        guard var writableKeys = currentKeys, var writableValues = currentValues else {
            fatalError("Quantized cache not properly initialized")
        }

        writableKeys.write(
            QuantizedKVState.quantizing(keys, groupSize: groupSize, bits: bits, mode: mode),
            range: plan.writeRange
        )
        writableValues.write(
            QuantizedKVState.quantizing(values, groupSize: groupSize, bits: bits, mode: mode),
            range: plan.writeRange
        )

        offset = plan.writeRange.upperBound
        self.keys = writableKeys
        self.values = writableValues

        return (
            writableKeys.prefix(upTo: offset).storage,
            writableValues.prefix(upTo: offset).storage
        )
    }

    /// This method is required by the KVCache protocol, but it is not intended to be used with QuantizedKVCache.
    /// Use `updateQuantized` instead.
    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            guard offset < keys.tokenCapacity else {
                return keys.arrays + values.arrays
            }
            let trimmedKeys = keys.prefix(upTo: offset)
            let trimmedValues = values.prefix(upTo: offset)
            return trimmedKeys.arrays + trimmedValues.arrays
        }
        set {
            switch newValue.count {
            case 4:
                keys = QuantizedKVState(weights: newValue[0], scales: newValue[1], biases: nil)
                values = QuantizedKVState(weights: newValue[2], scales: newValue[3], biases: nil)
            case 6:
                keys = QuantizedKVState(
                    weights: newValue[0],
                    scales: newValue[1],
                    biases: newValue[2]
                )
                values = QuantizedKVState(
                    weights: newValue[3],
                    scales: newValue[4],
                    biases: newValue[5]
                )
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }

            self.offset = Int(newValue[1]) ?? 0

            guard let storedStep = Int(newValue[0]),
                let storedGroupSize = Int(newValue[2]),
                let storedBits = Int(newValue[3])
            else {
                fatalError("Failed to convert QuantizedKVCache metaState values to integers")
            }
            guard storedStep == step, storedGroupSize == groupSize, storedBits == bits else {
                fatalError(
                    "QuantizedKVCache metaState does not match this cache configuration"
                )
            }
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    override func copy() -> KVCache {
        let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    /// Convert to unquantized cache
    internal func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            let currentKeys = keys.prefix(upTo: offset).storage
            let currentValues = values.prefix(upTo: offset).storage

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

/// Chunked KV cache for processing large contexts in chunks
internal class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    internal func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            offset - startPosition >= chunkSize
        else { return }

        let activeLength = offset - startPosition
        let trimCount = activeLength - chunkSize
        startPosition += trimCount
        guard let values else {
            replaceStorage(nil)
            return
        }
        replaceStorage(
            DenseKVStorage(
                keys: keys[.ellipsis, trimCount ..< activeLength, 0...],
                values: values[.ellipsis, trimCount ..< activeLength, 0...]
            )
        )
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        var storage = Self.storage(keys: self.keys, values: self.values)
        let plan = KVCacheAppendPlan(
            offset: offset,
            baseOffset: startPosition,
            incomingTokenCount: keys.dim(2),
            currentCapacity: storage?.capacity,
            step: step
        )

        if plan.needsGrowth {
            storage = appendDenseCapacity(to: storage, plan: plan, keys: keys, values: values)
        }

        guard var storage else {
            fatalError("Chunked KV cache storage was not initialized")
        }

        storage.write(keys: keys, values: values, range: plan.writeRange)
        replaceStorage(storage)
        offset += keys.dim(2)
        let end = offset - startPosition

        return (storage.keys[.ellipsis, ..<end, 0...], storage.values[.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }

    override func copy() -> KVCache {
        let new = ChunkedKVCache(chunkSize: chunkSize)
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }
}

/// Simple cache for Mamba-style state space models
internal class MambaCache: BaseKVCache {
    private var cache: [MLXArray?] = [nil, nil]

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        // Mamba doesn't use traditional KV cache update pattern
        fatalError("MambaCache should not use update(keys:values:) - use subscript access instead")
    }

    public func makeMask(N: Int) -> MLXArray? {
        nil
    }

    public override var state: [MLXArray] {
        get {
            // Empty arrays preserve nil slots without adding separate metadata.
            var result: [MLXArray] = []
            for item in cache {
                if let array = item {
                    result.append(array)
                } else {
                    result.append(MLXArray.zeros([0], dtype: .float32))
                }
            }
            return result
        }
        set {
            guard newValue.count == cache.count else {
                fatalError("MambaCache state must have exactly \(cache.count) elements")
            }
            for (i, array) in newValue.enumerated() {
                // Check if this is our nil placeholder (empty array with size 0)
                if array.size == 0 {
                    cache[i] = nil
                } else {
                    cache[i] = array
                }
            }
        }
    }

    override func copy() -> KVCache {
        let new = MambaCache()
        new.state = state.map { $0[.ellipsis] }
        return new
    }
}

/// Composite cache that manages multiple sub-caches
internal class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    public init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    internal var layoutCaches: [KVCache] {
        caches
    }

    @discardableResult
    internal func replaceLayoutCaches(
        _ transform: (KVCache) -> (KVCache, Int)
    ) -> Int {
        var convertedCount = 0
        for index in caches.indices {
            let result = transform(caches[index])
            caches[index] = result.0
            convertedCount += result.1
        }
        return convertedCount
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            guard stateLengths.reduce(0, +) == newValue.count else {
                fatalError("CacheList state does not match child cache layout")
            }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        guard !caches.isEmpty else {
            return 0
        }
        let trimmedCounts = caches.map { cache in
            cache.trim(n)
        }
        return trimmedCounts.min() ?? 0
    }

    override func copy() -> KVCache {
        let new = CacheList()
        new.caches = caches.map { $0.copy() }
        return new
    }
}

// MARK: - Error Types

struct KVCacheError: Error {
    let message: String
}

private struct KVCacheLayoutDescriptor: Codable, Equatable {
    let className: String
    let metaState: [String]
    let stateCount: Int
    let children: [KVCacheLayoutDescriptor]
}

// MARK: - Utility Functions

/// Save a pre-computed prompt cache to a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
///   - cache: The model cache state
///   - metadata: Optional metadata to save along with cache state
internal func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    let cacheClasses = cache.map(cacheClassName)
    let cacheLayouts = cache.map(cacheLayoutDescriptor)

    // Flatten cache data using the stable "i.j" tensor-key format.
    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    // cache_metadata is stored as [cache_info, user_metadata, cache_classes].
    var flattenedMetadata: [String: String] = [:]

    // Flatten cache_info as "0.i.j" (first element of cache_metadata)
    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    // Flatten user metadata as "1.key" (second element of cache_metadata)
    for (key, value) in metadata {
        flattenedMetadata["1.\(key)"] = value
    }

    // Flatten cache_classes as "2.i" (third element of cache_metadata)
    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    // Recursive layout metadata lets Swift round-trip composite caches while
    // preserving the existing top-level arrays/classes.
    for (i, layout) in cacheLayouts.enumerated() {
        let layoutData = try JSONEncoder().encode(layout)
        flattenedMetadata["3.\(i)"] = layoutData.base64EncodedString()
    }

    try save(arrays: flattenedData, metadata: flattenedMetadata, url: url)
}

/// Load a prompt cache from a file.
///
/// - Parameters:
///   - url: The URL to the `.safetensors` file
/// - Returns: The prompt cache and the metadata
internal func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]?) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)
    return try loadPromptCache(arrays: arrays, metadata: metadata)
}

/// Load a prompt cache from safetensors data already held in memory.
internal func loadPromptCache(
    data: Data
) throws -> ([KVCache], [String: String]?) {
    let (arrays, metadata) = try loadArraysAndMetadata(data: data)
    return try loadPromptCache(arrays: arrays, metadata: metadata)
}

private func loadPromptCache(
    arrays: [String: MLXArray],
    metadata: [String: String]
) throws -> ([KVCache], [String: String]?) {
    // Rebuild per-layer arrays from the flat tensor-key map.
    let cacheData = unflattenArrays(arrays)

    // Rebuild cache metadata from the flat metadata-key map.
    let unflattenedMetadata = unflattenMetadata(metadata)

    // Extract cache_info, user_metadata, and cache_classes from unflattened structure
    // Structure: [cache_info, user_metadata, cache_classes]
    guard unflattenedMetadata.count >= 3 else {
        throw KVCacheError(message: "Invalid cache metadata format")
    }

    var cacheInfo = unflattenedMetadata[0] as? [[String]] ?? []
    let userMetadata = unflattenedMetadata[1] as? [String: String] ?? [:]
    let cacheClasses = unflattenedMetadata[2] as? [String] ?? []
    let cacheLayouts = try unflattenCacheLayouts(metadata)

    if cacheInfo.count < cacheData.count {
        cacheInfo.append(contentsOf: Array(repeating: [], count: cacheData.count - cacheInfo.count))
    }

    guard cacheData.count == cacheInfo.count && cacheData.count == cacheClasses.count else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    // Reconstruct cache instances
    var caches: [KVCache] = []
    for i in 0 ..< cacheData.count {
        let cache: KVCache
        if let layout = cacheLayouts[i] {
            cache = try makeCache(layout: layout, state: cacheData[i])
        } else {
            cache = try makeCache(
                className: cacheClasses[i],
                metaState: cacheInfo[i],
                state: cacheData[i]
            )
        }
        caches.append(cache)
    }

    return (caches, userMetadata)
}

private func cacheClassName(_ cache: KVCache) -> String {
    // Keep stable class names so persisted caches survive implementation changes.
    switch cache {
    case is MiniMaxM3BatchKVCache:
        return "MiniMaxM3BatchKVCache"
    case is MiniMaxM3KVCache:
        return "MiniMaxM3KVCache"
    case is QuantizedRotatingKVCache:
        return "QuantizedRotatingKVCache"
    case is QuantizedKVCache:
        return "QuantizedKVCache"
    case is ChunkedKVCache:
        return "ChunkedKVCache"
    case is RotatingKVCache:
        return "RotatingKVCache"
    case is MambaCache:
        return "MambaCache"
    case is CacheList:
        return "CacheList"
    case is KVCacheSimple:
        return "KVCache"
    default:
        return "KVCache"
    }
}

private func cacheLayoutDescriptor(_ cache: KVCache) -> KVCacheLayoutDescriptor {
    let children: [KVCacheLayoutDescriptor]
    if let cacheList = cache as? CacheList {
        children = cacheList.layoutCaches.map(cacheLayoutDescriptor)
    } else {
        children = []
    }
    return KVCacheLayoutDescriptor(
        className: cacheClassName(cache),
        metaState: cache.metaState,
        stateCount: cache.state.count,
        children: children
    )
}

private func unflattenCacheLayouts(
    _ flatMetadata: [String: String]
) throws -> [Int: KVCacheLayoutDescriptor] {
    var layouts: [Int: KVCacheLayoutDescriptor] = [:]
    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")
        guard components.count == 2,
            components[0] == "3",
            let index = Int(components[1])
        else {
            continue
        }
        guard let data = Data(base64Encoded: value) else {
            throw KVCacheError(message: "Invalid cache layout metadata encoding")
        }
        layouts[index] = try JSONDecoder().decode(KVCacheLayoutDescriptor.self, from: data)
    }
    return layouts
}

private func makeCache(
    layout: KVCacheLayoutDescriptor,
    state: [MLXArray]
) throws -> KVCache {
    guard state.count == layout.stateCount else {
        throw KVCacheError(message: "Cache state count does not match layout metadata")
    }
    if layout.className == "CacheList" {
        return try makeCacheList(layout: layout, state: state)
    }
    return try makeLeafCache(
        className: layout.className,
        metaState: layout.metaState,
        state: state
    )
}

private func makeCacheList(
    layout: KVCacheLayoutDescriptor,
    state: [MLXArray]
) throws -> KVCache {
    guard !layout.children.isEmpty else {
        throw KVCacheError(message: "CacheList cache is missing child layout metadata")
    }

    var cursor = 0
    var children: [KVCache] = []
    children.reserveCapacity(layout.children.count)
    for childLayout in layout.children {
        let nextCursor = cursor + childLayout.stateCount
        guard nextCursor <= state.count else {
            throw KVCacheError(message: "CacheList child state exceeds parent state")
        }
        let childState = Array(state[cursor ..< nextCursor])
        children.append(try makeCache(layout: childLayout, state: childState))
        cursor = nextCursor
    }
    guard cursor == state.count else {
        throw KVCacheError(message: "CacheList layout did not consume all state arrays")
    }
    return CacheList(caches: children)
}

private func makeCache(
    className: String,
    metaState: [String],
    state: [MLXArray]
) throws -> KVCache {
    guard className != "CacheList" else {
        throw KVCacheError(message: "CacheList cache is missing layout metadata")
    }
    return try makeLeafCache(className: className, metaState: metaState, state: state)
}

private func makeLeafCache(
    className: String,
    metaState: [String],
    state: [MLXArray]
) throws -> KVCache {
    var cache = try makeEmptyLeafCache(className: className, metaState: metaState)
    if !state.isEmpty {
        cache.state = state
    }
    if !metaState.isEmpty {
        cache.metaState = metaState
    }
    return cache
}

private func makeEmptyLeafCache(
    className: String,
    metaState: [String]
) throws -> KVCache {
    switch className {
    case "KVCache", "KVCacheSimple":
        return KVCacheSimple()

    case "RotatingKVCache":
        guard metaState.count >= 5 else {
            throw KVCacheError(message: "Invalid RotatingKVCache metaState - expected 5 values")
        }
        guard metaState[1] != "None" else {
            throw KVCacheError(
                message:
                    "RotatingKVCache with maxSize=None is not supported. This cache was created with invalid parameters."
            )
        }
        guard let maxSize = Int(metaState[1]) else {
            throw KVCacheError(
                message: "Failed to parse RotatingKVCache maxSize from: \(metaState[1])"
            )
        }
        return RotatingKVCache(maxSize: maxSize)

    case "QuantizedRotatingKVCache":
        guard metaState.count == 8 else {
            throw KVCacheError(
                message: "Invalid QuantizedRotatingKVCache metaState - expected 8 values"
            )
        }
        guard let maxSize = Int(metaState[1]),
            let groupSize = Int(metaState[5]),
            let bits = Int(metaState[6]),
            let mode = QuantizationMode(rawValue: metaState[7])
        else {
            throw KVCacheError(message: "Failed to parse QuantizedRotatingKVCache metaState")
        }
        return QuantizedRotatingKVCache(
            maxSize: maxSize,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )

    case "QuantizedKVCache":
        guard metaState.count == 4 else {
            throw KVCacheError(message: "Invalid QuantizedKVCache metaState - expected 4 values")
        }
        guard let storedStep = Int(metaState[0]),
            let groupSize = Int(metaState[2]),
            let bits = Int(metaState[3])
        else {
            throw KVCacheError(message: "Failed to parse QuantizedKVCache metaState")
        }
        guard storedStep == 256 else {
            throw KVCacheError(message: "Unsupported QuantizedKVCache step: \(storedStep)")
        }
        return QuantizedKVCache(groupSize: groupSize, bits: bits)

    case "MiniMaxM3KVCache":
        return MiniMaxM3KVCache()

    case "MiniMaxM3BatchKVCache":
        return MiniMaxM3BatchKVCache()

    case "ChunkedKVCache":
        return ChunkedKVCache()

    case "MambaCache":
        return MambaCache()

    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

/// Unflatten arrays from the persisted key format, e.g. "0.1" or "1.0".
private func unflattenArrays(_ flatArrays: [String: MLXArray]) -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    // Parse all keys and organize by indices
    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        if components.count >= 2,
            let i = Int(components[0]),
            let j = Int(components[1])
        {
            if arrayMap[i] == nil {
                arrayMap[i] = [:]
            }
            arrayMap[i]![j] = array
        }
    }

    // Convert to ordered array structure
    var result: [[MLXArray]] = []
    let maxI = arrayMap.keys.max() ?? -1

    for i in 0 ... maxI {
        if let innerMap = arrayMap[i] {
            let maxJ = innerMap.keys.max() ?? -1
            var innerArray: [MLXArray] = []
            for j in 0 ... maxJ {
                if let array = innerMap[j] {
                    innerArray.append(array)
                }
            }
            result.append(innerArray)
        } else {
            result.append([])
        }
    }

    return result
}

/// Unflatten metadata from the persisted key format.
private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            // Cache info: "0.i.j" format
            if let i = Int(components[1]), let j = Int(components[2]) {
                // Ensure cacheInfo is large enough
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                // Ensure inner array is large enough
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            // User metadata: "1.key" format
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            // Cache classes: "2.i" format
            if let i = Int(components[1]) {
                // Ensure cacheClasses is large enough
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

/// Construct the model's cache for use when generating.
///
/// This function will defer the cache construction to the model if it has a
/// `newCache` method, otherwise it will make a default KV cache.
internal func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    // The model already conforms to LanguageModel which has newCache
    // If it also conforms to KVCacheDimensionProvider, the extension will provide the implementation
    return model.newCache(parameters: parameters)
}

/// Legacy function for backwards compatibility
internal func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

/// Fallback function to create cache when layer count is known
///
/// This function creates a default cache structure when the number of layers is known.
/// Use this when `makePromptCache` cannot determine the layer count automatically.
internal func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) -> [KVCache] {
    if let maxKVSize = maxKVSize {
        return (0 ..< numLayers).map { _ in
            RotatingKVCache(
                maxSize: maxKVSize,
                keep: GenerationConstants.rotatingCacheKeepTokens
            )
        }
    } else {
        return (0 ..< numLayers).map { _ in KVCacheSimple() }
    }
}

/// Check if model's cache can be trimmed.
internal func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

/// Trim the model's cache by the given number of tokens.
///
/// This function will trim the cache if possible (in-place) and return the
/// number of tokens that were trimmed.
@discardableResult
internal func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}

// MARK: - Type Aliases

/// Standard KV cache - alias to KVCacheSimple for compatibility
internal typealias StandardKVCache = KVCacheSimple

// MARK: - Quantized Attention Operations

internal func attentionMaskFillValue(dtype: DType) -> MLXArray {
    MLXArray(-Float.greatestFiniteMagnitude, dtype: dtype)
}

internal func attentionMaskDisabledValue(matching mask: MLXArray) -> MLXArray {
    if mask.dtype == .bool {
        return MLXArray(false, dtype: .bool)
    }
    return attentionMaskFillValue(dtype: mask.dtype)
}

internal func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    // Scale queries
    var scaledQueries = queries * scale

    // Handle GQA (Grouped Query Attention)
    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2.map { expandedDimensions($0, axis: -3) }
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2.map { expandedDimensions($0, axis: -3) }
        )
    }

    // Compute attention scores using quantized matmul
    var scores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits, mode: mode
    )

    // Apply mask
    switch mask {
    case .causal:
        let (qL, kL) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kL - qL)
        let kIndices = MLXArray(0 ..< kL)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, attentionMaskFillValue(dtype: scores.dtype))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, attentionMaskFillValue(dtype: scores.dtype))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        // Handle multiple mask arrays - just use the first one for simplicity
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, attentionMaskFillValue(dtype: scores.dtype))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    let attentionWeights = softmax(scores, axis: -1)

    // Compute output using quantized matmul
    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits, mode: mode
    )

    // Reshape output for GQA
    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output
}

// MARK: - Dynamic Cache Quantization

/// Dynamically quantize KV caches during generation if conditions are met
///
/// Converts regular caches to quantized caches when:
/// - kvBits is specified
/// - The cache is not already quantized
/// - The cache offset is greater than quantizedKVStart
///
/// - Parameters:
///   - cache: Array of KV caches to potentially quantize
///   - kvBits: Number of bits for quantization (nil = no quantization)
///   - kvGroupSize: Group size for quantization
///   - quantizedKVStart: Token count threshold to begin quantizing
///   - skipLastLayer: Keep the final top-level layer cache in full precision.
internal func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0,
    skipLastLayer: Bool = false
) {
    guard let kvBits = kvBits,
        !cache.isEmpty,
        effectiveKVCacheOffset(cache[0]) > quantizedKVStart
    else {
        return
    }

    var convertedCount = 0
    let quantizedLayerCount = skipLastLayer ? max(cache.count - 1, 0) : cache.count
    for i in 0 ..< quantizedLayerCount {
        let result = quantizedKVCache(cache[i], groupSize: kvGroupSize, bits: kvBits)
        cache[i] = result.cache
        convertedCount += result.convertedCount
    }

    guard convertedCount > 0 else { return }
    MLXGenerationDiagnostics.recordQuantizedKVConversion(
        offset: effectiveKVCacheOffset(cache[0]),
        kvBits: kvBits,
        kvGroupSize: kvGroupSize,
        quantizedKVStart: quantizedKVStart,
        quantizedKVSkipLastLayer: skipLastLayer,
        convertedCount: convertedCount
    )
}

private func quantizedKVCache(
    _ cache: KVCache,
    groupSize: Int,
    bits: Int
) -> (cache: KVCache, convertedCount: Int) {
    switch cache {
    case is QuantizedKVCacheProtocol:
        return (cache, 0)
    case let simpleCache as KVCacheSimple:
        return (simpleCache.toQuantized(groupSize: groupSize, bits: bits), 1)
    case let rotatingCache as RotatingKVCache:
        return (rotatingCache.toQuantized(groupSize: groupSize, bits: bits), 1)
    case let cacheList as CacheList:
        let convertedCount = cacheList.replaceLayoutCaches { childCache in
            quantizedKVCache(childCache, groupSize: groupSize, bits: bits)
        }
        return (cacheList, convertedCount)
    default:
        return (cache, 0)
    }
}

private func effectiveKVCacheOffset(_ cache: KVCache) -> Int {
    guard let cacheList = cache as? CacheList else {
        return cache.offset
    }
    return cacheList.layoutCaches.map(effectiveKVCacheOffset).max() ?? cache.offset
}
