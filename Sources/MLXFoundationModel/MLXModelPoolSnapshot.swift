/// Observable ``MLXModelPool`` state for diagnostics and tests.
public struct MLXModelPoolSnapshot: Equatable, Sendable {
    /// All registered canonical model identifiers.
    public let registeredModelIDs: [String]

    /// Alias-to-canonical-model mapping.
    public let aliasTargets: [String: String]

    /// API-visible serving profile identifier to canonical model mapping.
    public let servingProfileTargets: [String: String]

    /// Base models and serving profile variants visible to API callers.
    public let visibleModels: [MLXModelPoolVisibleModel]

    /// Canonical model identifiers for resident sessions.
    public let residentModelIDs: [String]

    /// Resident model identifiers protected from eviction by pinning.
    public let pinnedResidentModelIDs: [String]

    /// Resident model identifiers with active scoped leases.
    public let leasedResidentModelIDs: [String]

    /// Resident model identifiers that will unload when their active lease drains.
    public let pendingUnloadResidentModelIDs: [String]

    /// Estimated resident model weight bytes by canonical model identifier.
    public let residentMemoryBytesByModelID: [String: Int]

    /// Total estimated resident model weight bytes.
    public let residentMemoryBytes: Int

    /// Creates a diagnostic snapshot.
    public init(
        registeredModelIDs: [String],
        aliasTargets: [String: String],
        residentModelIDs: [String],
        pinnedResidentModelIDs: [String],
        leasedResidentModelIDs: [String],
        pendingUnloadResidentModelIDs: [String] = [],
        residentMemoryBytesByModelID: [String: Int] = [:],
        servingProfileTargets: [String: String] = [:],
        visibleModels: [MLXModelPoolVisibleModel] = []
    ) {
        self.registeredModelIDs = registeredModelIDs
        self.aliasTargets = aliasTargets
        self.servingProfileTargets = servingProfileTargets
        self.visibleModels = visibleModels
        self.residentModelIDs = residentModelIDs
        self.pinnedResidentModelIDs = pinnedResidentModelIDs
        self.leasedResidentModelIDs = leasedResidentModelIDs
        self.pendingUnloadResidentModelIDs = pendingUnloadResidentModelIDs
        self.residentMemoryBytesByModelID = residentMemoryBytesByModelID
        self.residentMemoryBytes = residentMemoryBytesByModelID.values.reduce(0, +)
    }
}
