//
//  ModelAdapterTypeRegistry.swift
//  mlx-libraries
//
//  Created by Ivan Petrukha on 06.06.2025.
//

import Foundation

/// Thread-safe registry for managing model adapter creators (LoRA, DoRA, etc.)
///
/// This class is marked `@unchecked Sendable` because:
/// - It contains mutable state (`creators` dictionary) that is protected by explicit synchronization (`NSLock`)
/// - All mutations are guarded by lock-protected critical sections
/// - The lock ensures that all dictionary accesses are properly serialized
///
/// Safety guarantees:
/// - Atomic operations: All reads and writes to `creators` are protected by NSLock
/// - Thread-safe access: Multiple threads can safely register and retrieve adapter creators
/// - No data races: The lock serializes all access to the mutable dictionary
/// - Small critical sections: Lock contention is minimal due to short operations
internal class ModelAdapterTypeRegistry: @unchecked Sendable {

    /// Creates an empty registry.
    public init() {
        self.creators = [:]
    }

    /// Creates a registry with given creators.
    public init(creators: [String: @Sendable (URL) throws -> any ModelAdapter]) {
        self.creators = creators
    }

    // Note: using NSLock as we have very small (just dictionary get/set)
    // critical sections and expect no contention.  this allows the methods
    // to remain synchronous.
    private let lock = NSLock()
    private var creators: [String: @Sendable (URL) throws -> any ModelAdapter]

    /// Add a new model adapter to the type registry.
    internal func registerAdapterType(
        _ type: String, creator: @Sendable @escaping (URL) throws -> any ModelAdapter
    ) {
        lock.withLock {
            creators[type] = creator
        }
    }

    internal func createAdapter(directory: URL, adapterType: String) throws -> ModelAdapter {
        let creator = lock.withLock {
            creators[adapterType]
        }
        guard let creator else {
            throw ModelAdapterError.unsupportedAdapterType(adapterType)
        }
        return try creator(directory)
    }
}
