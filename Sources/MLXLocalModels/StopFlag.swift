import Foundation

/// Thread-safe stop flag that can be shared between actor and non-actor contexts
///
/// This class provides a thread-safe boolean flag for signaling cancellation
/// or stop conditions across concurrent contexts. It uses NSLock internally
/// to ensure atomic read and write operations.
///
/// # Thread Safety
/// All operations (`get()`, `set()`, `reset()`) are guaranteed to be atomic
/// and thread-safe. Multiple threads can safely read and write to this flag
/// concurrently without data races.
///
/// The class is marked `@unchecked Sendable` because it contains mutable state
/// (`_value`) that is protected by explicit synchronization (`NSLock`). The lock
/// ensures that all accesses to `_value` are properly serialized.
///
/// # Performance
/// The `get()` operation is marked with `@inline(__always)` to minimize
/// overhead for frequent polling scenarios. Lock contention is minimal
/// as the critical sections are extremely short.
///
/// # Usage
/// ```swift
/// let stopFlag = StopFlag()
///
/// // Writer thread/actor
/// stopFlag.set(true)
///
/// // Reader thread (can be polled frequently)
/// if stopFlag.get() {
///     // Handle stop condition
/// }
///
/// // Reset for reuse
/// stopFlag.reset()
/// ```
internal final class StopFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: Bool = false

    internal init() {
        lock.name = "com.think.mlxsession.stopflag"
    }

    /// Atomically reads the current value of the stop flag
    ///
    /// This operation is thread-safe and can be called from any context.
    /// The `@inline(__always)` attribute minimizes overhead for frequent polling.
    ///
    /// - Returns: The current boolean value of the stop flag
    @inline(__always)
    internal func get() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return _value
    }

    /// Atomically sets the stop flag to a new value
    ///
    /// This operation is thread-safe and can be called from any context.
    ///
    /// - Parameter newValue: The new boolean value to set
    internal func set(_ newValue: Bool) {
        lock.lock()
        defer { lock.unlock() }
        _value = newValue
    }

    /// Atomically resets the stop flag to false
    ///
    /// This is equivalent to calling `set(false)` but provides
    /// a more explicit API for the common reset operation.
    internal func reset() {
        set(false)
    }

    deinit {
        // NSLock does not require explicit cleanup
        // The lock will be automatically deallocated
    }
}
