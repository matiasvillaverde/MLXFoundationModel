// Copyright © 2024 Apple Inc.

import Foundation

/// Thread-safe registry for managing language model type creators (Llama, Phi, Gemma, etc.)
///
/// This class is marked `@unchecked Sendable` because:
/// - It contains mutable state (`creators` dictionary) that is protected by explicit synchronization (`NSLock`)
/// - All mutations are guarded by lock-protected critical sections
/// - The lock ensures that all dictionary accesses are properly serialized
///
/// Safety guarantees:
/// - Atomic operations: All reads and writes to `creators` are protected by NSLock
/// - Thread-safe registration: Multiple threads can safely register model creators
/// - Thread-safe creation: Model instantiation from the registry is thread-safe
/// - No data races: The lock serializes all access to the mutable dictionary
/// - Small critical sections: Lock contention is minimal due to short operations
internal class ModelTypeRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    internal init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    internal init(creators: [String: @Sendable (URL) throws -> any LanguageModel]) {
        self.creators = creators
    }

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention. this allows the methods
    // to remain synchronous.
    private let lock = NSLock()
    private var creators: [String: @Sendable (URL) throws -> any LanguageModel]

    /// Add a new model to the type registry.
    internal func registerModelType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any LanguageModel
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    /// Given a `modelType` and configuration file instantiate a new `LanguageModel`.
    internal func createModel(configuration: URL, modelType: String) throws -> LanguageModel {
        let creator = lock.withLock {
            creators[modelType]
        }
        guard let creator else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        return try creator(configuration)
    }

    internal func registeredModelTypes() -> [String] {
        lock.withLock {
            creators.keys.sorted()
        }
    }
}
