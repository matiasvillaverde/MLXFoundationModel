// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Container for models that guarantees single threaded access.
///
/// Wrap models used by e.g. the UI in a ModelContainer. Callers can access
/// the model and/or tokenizer (any values from the ``ModelContext``):
///
/// ```swift
/// let messages = [["role": "user", "content": prompt]]
/// let promptTokens = try await modelContainer.perform { context in
///     try context.tokenizer.applyChatTemplate(messages: messages)
/// }
/// ```
///
/// or:
///
/// ```swift
/// let prompt: String
/// let result = await modelContainer.perform { context in
///     let input = try context.tokenize(prompt: prompt)
///     return generate(
///         input: input, parameters: generateParameters, context: context
///     ) { tokens in
///     ...
///     }
/// }
/// ```
internal actor ModelContainer {
    var context: ModelContext
    internal var configuration: ModelConfiguration { context.configuration }
    private var promptCacheEntries: [PromptCacheEntry] = []

    internal init(context: ModelContext) {
        self.context = context
    }

    /// Perform an action on the model and/or tokenizer. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(_:) that uses a ModelContext")
    internal func perform<R>(_ action: @Sendable (any LanguageModel, Tokenizer) throws -> R) rethrows
        -> R
    {
        try action(context.model, context.tokenizer)
    }

    /// Perform an action on the model and/or tokenizer with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    @available(*, deprecated, message: "prefer perform(values:_:) that uses a ModelContext")
    internal func perform<V, R>(
        values: V, _ action: @Sendable (any LanguageModel, Tokenizer, V) throws -> R
    ) rethrows -> R {
        try action(context.model, context.tokenizer, values)
    }

    /// Perform an action on the ``ModelContext``. Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    internal func perform<R>(_ action: @Sendable (ModelContext) async throws -> sending R) async rethrows -> R {
        try await action(context)
    }

    /// Perform an action on the ``ModelContext`` with additional context values.
    /// Callers _must_ eval any `MLXArray` before returning as
    /// `MLXArray` is not `Sendable`.
    internal func perform<V: Sendable, R>(
        values: V, _ action: @Sendable (ModelContext, V) async throws -> sending R
    ) async rethrows -> R {
        try await action(context, values)
    }

    /// Perform generation with access to the model-owned prompt cache.
    internal func performWithPromptCache<R>(
        _ action: @Sendable (ModelContext, inout [PromptCacheEntry]) throws -> sending R
    ) rethrows -> R {
        try action(context, &promptCacheEntries)
    }

    /// Update the owned `ModelContext`.
    /// - Parameter action: update action
    internal func update(_ action: @Sendable (inout ModelContext) -> Void) {
        action(&context)
    }

    internal func clearPromptCache() {
        promptCacheEntries.removeAll()
    }

    internal func promptCacheEntriesSnapshot() -> [PromptCacheEntry] {
        promptCacheEntries
    }

    internal func replacePromptCacheEntries(_ entries: [PromptCacheEntry]) {
        promptCacheEntries = entries
    }
}
