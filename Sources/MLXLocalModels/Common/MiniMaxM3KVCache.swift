import MLX

internal final class MiniMaxM3KVCache: BaseKVCache, CustomDebugStringConvertible {
    private var kvCache = KVCacheSimple()
    private var indexKeys: MLXArray?
    private var indexOffset = 0

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let updated = kvCache.update(keys: keys, values: values)
        offset = kvCache.offset
        return updated
    }

    @discardableResult
    internal func updateIndex(keys: MLXArray) -> MLXArray {
        if let current = indexKeys {
            indexKeys = concatenated([current[.ellipsis, ..<indexOffset, 0...], keys], axis: 2)
        } else {
            indexKeys = keys
        }
        indexOffset += keys.dim(2)
        return indexKeys![.ellipsis, ..<indexOffset, 0...]
    }

    public override var state: [MLXArray] {
        get {
            let kvState = kvCache.state
            guard kvState.count == 2 else {
                return []
            }
            guard let indexKeys else {
                return kvState
            }
            let indexState = indexKeys[.ellipsis, ..<min(indexOffset, indexKeys.dim(2)), 0...]
            return kvState + [indexState]
        }
        set {
            guard newValue.count == 2 || newValue.count == 3 else {
                fatalError("MiniMaxM3KVCache state must have keys, values, and optional index keys")
            }
            kvCache.state = Array(newValue.prefix(2))
            offset = kvCache.offset
            indexKeys = newValue.count == 3 ? newValue[2] : nil
            indexOffset = indexKeys?.dim(2) ?? 0
        }
    }

    public override var metaState: [String] {
        get { [String(indexOffset)] }
        set {
            guard newValue.count == 1, let parsed = Int(newValue[0]) else {
                fatalError("MiniMaxM3KVCache metaState must contain one integer index offset")
            }
            indexOffset = indexKeys?.dim(2) ?? parsed
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = kvCache.trim(n)
        offset = kvCache.offset
        indexOffset = max(0, indexOffset - trimmed)
        return trimmed
    }

    public override func makeMask(
        n: Int,
        windowSize: Int?,
        returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        kvCache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    override func copy() -> KVCache {
        let new = MiniMaxM3KVCache()
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    internal var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), indexOffset: \(indexOffset)"
    }
}

internal final class MiniMaxM3BatchKVCache: BaseKVCache, CustomDebugStringConvertible {
    private var keys: MLXArray?
    private var values: MLXArray?
    private var offsets: MLXArray?
    private var leftPadding: MLXArray?
    private var indexKeys: MLXArray?
    private var indexOffset = 0

    internal init(leftPadding: MLXArray? = nil) {
        self.leftPadding = leftPadding
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("MiniMaxM3BatchKVCache is a serialized batched cache and cannot update directly")
    }

    public override var state: [MLXArray] {
        get {
            [keys, values, offsets, leftPadding, indexKeys].compactMap(\.self)
        }
        set {
            guard newValue.count == 5 else {
                fatalError("MiniMaxM3BatchKVCache state must contain keys, values, offsets, left padding, and index keys")
            }
            keys = newValue[0]
            values = newValue[1]
            offsets = newValue[2]
            leftPadding = newValue[3]
            indexKeys = newValue[4]
            offset = keys?.dim(2) ?? indexKeys?.dim(2) ?? 0
            indexOffset = indexKeys?.dim(2) ?? 0
        }
    }

    public override var metaState: [String] {
        get { [String(indexOffset)] }
        set {
            guard newValue.count == 1, let parsed = Int(newValue[0]) else {
                fatalError("MiniMaxM3BatchKVCache metaState must contain one integer index offset")
            }
            indexOffset = indexKeys?.dim(2) ?? parsed
        }
    }

    public override var isTrimmable: Bool { false }

    override func copy() -> KVCache {
        let new = MiniMaxM3BatchKVCache()
        let currentState = state
        if !currentState.isEmpty {
            new.state = currentState.map { $0[.ellipsis] }
        }
        new.metaState = metaState
        return new
    }

    internal func extract(_ row: Int) -> MiniMaxM3KVCache {
        guard let keys, let values, let offsets, let leftPadding, let indexKeys else {
            fatalError("MiniMaxM3BatchKVCache cannot extract from empty state")
        }
        let offsetValues = Self.intValues(offsets)
        let paddingValues = Self.intValues(leftPadding)
        guard row >= 0, row < offsetValues.count, row < paddingValues.count else {
            fatalError("MiniMaxM3BatchKVCache row is out of bounds")
        }

        let start = paddingValues[row]
        let end = start + offsetValues[row]
        let cache = MiniMaxM3KVCache()
        cache.state = [
            keys[row ..< row + 1, 0..., start ..< end, 0...],
            values[row ..< row + 1, 0..., start ..< end, 0...],
            indexKeys[row ..< row + 1, 0..., start ..< end, 0...],
        ]
        return cache
    }

    internal static func merge(_ caches: [MiniMaxM3KVCache]) -> MiniMaxM3BatchKVCache {
        guard !caches.isEmpty else {
            return MiniMaxM3BatchKVCache()
        }
        let states = caches.map(\.state)
        guard states.allSatisfy({ $0.count == 3 }) else {
            fatalError("MiniMaxM3BatchKVCache requires sparse index state for every row")
        }
        let lengths = states.map { $0[0].dim(2) }
        let maxLength = lengths.max() ?? 0
        let leftPaddingValues = lengths.map { maxLength - $0 }
        let batch = MiniMaxM3BatchKVCache()
        batch.state = [
            concatenated(zip(states, leftPaddingValues).map { pair in
                Self.leftPad(pair.0[0], count: pair.1)
            }, axis: 0),
            concatenated(zip(states, leftPaddingValues).map { pair in
                Self.leftPad(pair.0[1], count: pair.1)
            }, axis: 0),
            MLXArray(lengths.map(Int32.init)),
            MLXArray(leftPaddingValues.map(Int32.init)),
            concatenated(zip(states, leftPaddingValues).map { pair in
                Self.leftPad(pair.0[2], count: pair.1)
            }, axis: 0),
        ]
        return batch
    }

    internal var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), indexOffset: \(indexOffset)"
    }

    private static func leftPad(_ array: MLXArray, count: Int) -> MLXArray {
        guard count > 0 else {
            return array
        }
        let shape = [array.dim(0), array.dim(1), count, array.dim(3)]
        let padding = MLXArray.zeros(shape, dtype: array.dtype)
        return concatenated([padding, array], axis: 2)
    }

    private static func intValues(_ array: MLXArray) -> [Int] {
        eval(array)
        if array.dtype == .int32 {
            return array.asArray(Int32.self).map(Int.init)
        }
        return array.asArray(Int.self)
    }
}
