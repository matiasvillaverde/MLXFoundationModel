/// Pool-wide residency limits for ``MLXModelPool``.
public struct MLXModelPoolConfiguration: Equatable, Hashable, Sendable {
    /// Maximum number of model sessions allowed to stay resident.
    public let maxResidentModels: Int

    /// Optional maximum estimated resident model weight bytes.
    public let maxResidentMemoryBytes: Int?

    /// Creates a pool configuration.
    ///
    /// - Parameter maxResidentModels: Maximum resident session count. Values
    ///   below one are normalized to one.
    /// - Parameter maxResidentMemoryBytes: Optional estimated resident weight
    ///   byte budget. Non-positive values disable byte-based admission.
    public init(
        maxResidentModels: Int = 1,
        maxResidentMemoryBytes: Int? = nil
    ) {
        self.maxResidentModels = max(1, maxResidentModels)
        self.maxResidentMemoryBytes = maxResidentMemoryBytes.flatMap { bytes in
            bytes > 0 ? bytes : nil
        }
    }
}
