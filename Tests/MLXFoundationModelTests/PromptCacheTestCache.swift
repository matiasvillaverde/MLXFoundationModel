@testable import MLXLocalModels

final class PromptCacheTestCache: BaseKVCache {
    init(offset: Int) {
        super.init()
        self.offset = offset
    }

    deinit {
        // Required by the strict test-target lint profile.
    }

    override var isTrimmable: Bool { true }

    @discardableResult
    override func trim(_ tokenCount: Int) -> Int {
        let trimmed = min(offset, tokenCount)
        offset -= trimmed
        return trimmed
    }

    override func copy() -> KVCache {
        PromptCacheTestCache(offset: offset)
    }
}
