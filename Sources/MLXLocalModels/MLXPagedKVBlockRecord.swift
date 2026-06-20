internal struct MLXPagedKVBlockRecord: Sendable, Equatable {
    internal let id: MLXPagedKVBlockID
    internal var blockHash: String
    internal var tokenCount: Int
    internal var refCount: Int
    internal var lastAccessTick: UInt64

    internal var isEvictable: Bool {
        refCount == 0
    }
}
