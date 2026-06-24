import Tokenizers

/// Actor-owned model state used to keep model, tokenizer, and prompt-cache access serialized.
internal actor ModelContainer {
    internal var context: ModelContext
    internal var configuration: ModelConfiguration { context.configuration }

    private var promptCacheEntries: [PromptCacheEntry] = []

    internal init(context: ModelContext) {
        self.context = context
    }

    /// Runs an action with the model and tokenizer.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    internal func perform<R>(_ action: @Sendable (any LanguageModel, Tokenizer) throws -> R) rethrows
        -> R
    {
        try action(context.model, context.tokenizer)
    }

    /// Runs an action with the model, tokenizer, and caller-provided values.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    internal func perform<V, R>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> R
    ) rethrows -> R {
        try action(context.model, context.tokenizer, values)
    }

    /// Runs an action with the current model context.
    internal func perform<R>(_ action: @Sendable (ModelContext) async throws -> sending R) async rethrows -> R {
        try await action(context)
    }

    /// Runs an action with the current model context and caller-provided values.
    internal func perform<V: Sendable, R>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> sending R
    ) async rethrows -> R {
        try await action(context, values)
    }

    /// Runs an action with mutable access to the prompt cache owned by this model.
    internal func performWithPromptCache<R>(
        _ action: @Sendable (ModelContext, inout [PromptCacheEntry]) throws -> sending R
    ) rethrows -> R {
        try action(context, &promptCacheEntries)
    }

    /// Mutates the model context without exposing writable actor state.
    internal func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }

    internal func clearPromptCache() {
        promptCacheEntries.removeAll(keepingCapacity: true)
    }

    internal func promptCacheEntriesSnapshot() -> [PromptCacheEntry] {
        promptCacheEntries
    }

    internal func replacePromptCacheEntries(_ entries: [PromptCacheEntry]) {
        guard !entries.isEmpty else {
            clearPromptCache()
            return
        }
        promptCacheEntries = entries
    }
}
