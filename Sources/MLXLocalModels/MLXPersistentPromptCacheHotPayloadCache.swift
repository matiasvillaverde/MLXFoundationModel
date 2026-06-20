import Foundation

internal struct MLXPersistentPromptCacheHotPayloadSnapshot: Equatable, Sendable {
    let maxBytes: Int
    let totalBytes: Int
    let entryCount: Int
    let keys: [String]

    var availableBytes: Int {
        max(maxBytes - totalBytes, 0)
    }
}

internal final class MLXPersistentPromptCacheHotPayloadCache: @unchecked Sendable {
    private struct Entry {
        let data: Data
        let byteCount: Int
        var accessCounter: UInt64
    }

    private let lock = NSLock()
    private var maxBytes: Int
    private var totalBytes = 0
    private var accessCounter: UInt64 = 0
    private var entries: [String: Entry] = [:]

    internal init(maxBytes: Int) {
        self.maxBytes = max(0, maxBytes)
        lock.name = "org.mlxfoundationmodel.persistent-cache-hot-payload"
    }

    internal func configure(maxBytes: Int) -> Int {
        lock.lock()
        self.maxBytes = max(0, maxBytes)
        let evictedCount = pruneIfNeeded()
        lock.unlock()
        return evictedCount
    }

    internal func data(forKey key: String) -> Data? {
        lock.lock()
        defer { lock.unlock() }
        guard var entry = entries[key] else {
            return nil
        }
        accessCounter &+= 1
        entry.accessCounter = accessCounter
        entries[key] = entry
        return entry.data
    }

    internal func store(_ data: Data, forKey key: String) -> Int {
        lock.lock()
        defer { lock.unlock() }

        removeValueWithoutLock(forKey: key)
        guard maxBytes > 0, data.count <= maxBytes else {
            return 0
        }

        accessCounter &+= 1
        entries[key] = Entry(
            data: data,
            byteCount: data.count,
            accessCounter: accessCounter
        )
        totalBytes += data.count
        return pruneIfNeeded()
    }

    internal func storeIfFits(_ data: Data, forKey key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }

        let existingByteCount = entries[key]?.byteCount ?? 0
        let availableBytes = maxBytes - totalBytes + existingByteCount
        guard maxBytes > 0, data.count <= availableBytes else {
            return false
        }

        removeValueWithoutLock(forKey: key)
        accessCounter &+= 1
        entries[key] = Entry(
            data: data,
            byteCount: data.count,
            accessCounter: accessCounter
        )
        totalBytes += data.count
        return true
    }

    @discardableResult
    internal func removeValue(forKey key: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return removeValueWithoutLock(forKey: key)
    }

    internal func removeAll() {
        lock.lock()
        entries.removeAll(keepingCapacity: true)
        totalBytes = 0
        lock.unlock()
    }

    internal func snapshot() -> MLXPersistentPromptCacheHotPayloadSnapshot {
        lock.lock()
        let snapshot = MLXPersistentPromptCacheHotPayloadSnapshot(
            maxBytes: maxBytes,
            totalBytes: totalBytes,
            entryCount: entries.count,
            keys: entries.keys.sorted()
        )
        lock.unlock()
        return snapshot
    }

    private func pruneIfNeeded() -> Int {
        var evictedCount = 0
        while totalBytes > maxBytes,
            let key = entries.min(by: { lhs, rhs in
                lhs.value.accessCounter < rhs.value.accessCounter
            })?.key {
            if removeValueWithoutLock(forKey: key) {
                evictedCount += 1
            }
        }
        return evictedCount
    }

    @discardableResult
    private func removeValueWithoutLock(forKey key: String) -> Bool {
        guard let removed = entries.removeValue(forKey: key) else {
            return false
        }
        totalBytes -= removed.byteCount
        return true
    }
}
