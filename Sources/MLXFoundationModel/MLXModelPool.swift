import MLXLocalModels

/// Actor-backed residency pool for MLX Foundation Models.
///
/// The pool centralizes server-style residency policy for MLX-backed
/// Foundation Models: aliases, preloading, warm reuse, cold unload after use,
/// pinned model protection, idle TTL, and LRU eviction.
public actor MLXModelPool {
    /// Factory used to create a fresh MLX generating session for a resident model.
    public typealias SessionFactory = MLXModelPoolSessionFactory

    /// Pool-wide residency limits.
    public typealias Configuration = MLXModelPoolConfiguration

    /// Observable pool state for tests, diagnostics, and host dashboards.
    public typealias Snapshot = MLXModelPoolSnapshot

    let configuration: Configuration
    let sessionFactory: SessionFactory
    var registrations: [String: MLXLanguageModel] = [:]
    var aliasTargets: [String: String] = [:]
    var servingProfiles: [String: MLXModelServingProfile] = [:]
    var servingProfileTargets: [String: String] = [:]
    var residents: [ProviderConfiguration: MLXModelPoolResidentEntry] = [:]
    var loadingResidents: [ProviderConfiguration: MLXModelPoolLoadingEntry] = [:]

    /// Creates a model pool.
    ///
    /// - Parameters:
    ///   - configuration: Residency limits for loaded sessions.
    ///   - sessionFactory: Factory used when a model first becomes resident.
    public init(
        configuration: Configuration = Configuration(),
        sessionFactory: @escaping SessionFactory = { MLXSessionFactory.create() }
    ) {
        self.configuration = configuration
        self.sessionFactory = sessionFactory
    }
}
