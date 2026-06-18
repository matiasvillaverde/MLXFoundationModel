// Copyright © 2024 Apple Inc.

import Foundation

/// Thread-safe registry for managing model configurations
///
/// This class is marked `@unchecked Sendable` because:
/// - It contains mutable state (`registry`) that is protected by explicit synchronization (`NSLock`)
/// - All mutations are guarded by lock-protected critical sections
/// - The lock ensures that all dictionary accesses are properly serialized
///
/// Safety guarantees:
/// - Atomic operations: All reads and writes to `registry` are protected by NSLock
/// - Thread-safe access: Multiple threads can safely query and modify the registry
/// - No data races: The lock serializes all access to the mutable dictionary
internal class AbstractModelRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    internal init() {
        self.registry = Dictionary()
    }

    /// Creates a new registry with from given model configurations.
    internal init(modelConfigurations: [ModelConfiguration]) {
        self.registry = Dictionary(uniqueKeysWithValues: modelConfigurations.map { ($0.name, $0) })
    }

    private let lock = NSLock()
    private var registry: [String: ModelConfiguration]

    internal func register(configurations: [ModelConfiguration]) {
        lock.withLock {
            for c in configurations {
                registry[c.name] = c
            }
        }
    }

    /// Returns configuration from ``modelRegistry``.
    ///
    /// - Note: If the id doesn't exists in the configuration, this will return a new instance of it.
    /// If you want to check if the configuration in model registry, you should use ``contains(id:)``.
    internal func configuration(id: String) -> ModelConfiguration {
        lock.withLock {
            if let c = registry[id] {
                return c
            } else {
                return ModelConfiguration(id: id)
            }
        }
    }

    /// Returns true if the registry contains a model with the id. Otherwise, false.
    internal func contains(id: String) -> Bool {
        lock.withLock {
            registry[id] != nil
        }
    }

    internal var models: some Collection<ModelConfiguration> & Sendable {
        lock.withLock {
            return registry.values
        }
    }
}
