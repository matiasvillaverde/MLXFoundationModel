@testable import MLXLocalModels

enum PromptCacheTestIsolation {
    private actor Lock {
        private var isLocked = false
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func acquire() async {
            if isLocked {
                await withCheckedContinuation { continuation in
                    waiters.append(continuation)
                }
            } else {
                isLocked = true
            }
        }

        func release() {
            guard !waiters.isEmpty else {
                isLocked = false
                return
            }
            waiters.removeFirst().resume()
        }
    }

    private static let lock = Lock()

    static func withLock<T>(_ operation: () async throws -> T) async throws -> T {
        await lock.acquire()
        do {
            let result = try await operation()
            await lock.release()
            return result
        } catch {
            await lock.release()
            throw error
        }
    }

    static func resetSharedHotCache() {
        MLXPersistentPromptCacheBlockStore.clearHotCache()
        MLXPersistentPromptCacheBlockStore.clearTipLineage()
        MLXPersistentPromptCacheBlockStore.configureHotCache(limitBytes: 67_108_864)
    }
}
