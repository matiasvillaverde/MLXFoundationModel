import Foundation
import MLXLocalModels

extension MLXModelPool {
    /// Preloads a registered model and keeps its session resident.
    @discardableResult
    public func preload(
        id: String,
        now: Date = Date()
    ) async throws -> any MLXGeneratingSession {
        let model = try model(id: id)
        return try await session(for: model, now: now)
    }

    /// Returns a resident session for the supplied model, loading it if needed.
    @discardableResult
    public func session(
        for model: MLXLanguageModel,
        now: Date = Date()
    ) async throws -> any MLXGeneratingSession {
        try await residentSession(for: model, now: now)
    }

    /// Runs an operation with a scoped session lease.
    ///
    /// Leased sessions are protected from LRU eviction until the operation
    /// returns or throws. Models with cold residency unload after the final
    /// lease is released.
    @preconcurrency
    public func withSession<Result: Sendable>(
        id: String,
        now: Date = Date(),
        _ operation: @Sendable (any MLXGeneratingSession) async throws -> Result
    ) async throws -> Result {
        let model = try model(id: id)
        return try await withSession(for: model, now: now, operation)
    }

    /// Runs an operation with a scoped session lease for an explicit model.
    ///
    /// This overload does not require prior registration and is useful for
    /// adapter layers that already hold a fully configured ``MLXLanguageModel``.
    @preconcurrency
    public func withSession<Result: Sendable>(
        for model: MLXLanguageModel,
        now: Date = Date(),
        _ operation: @Sendable (any MLXGeneratingSession) async throws -> Result
    ) async throws -> Result {
        let key = model.providerConfiguration
        let session = try await leaseSession(for: model, now: now)
        do {
            let result = try await operation(session)
            await releaseLease(for: key, now: Date())
            return result
        } catch {
            await releaseLease(for: key, now: Date())
            throw error
        }
    }

    /// Streams generation through a leased resident session.
    ///
    /// The lease is held until the returned stream finishes, fails, or is
    /// cancelled by the consumer.
    nonisolated public func stream(
        for model: MLXLanguageModel,
        input: LLMInput,
        now: Date = Date()
    ) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await withSession(for: model, now: now) { session in
                        let source = await session.stream(input)
                        for try await chunk in source {
                            continuation.yield(chunk)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }

    /// Unloads or schedules unload for unpinned resident sessions for the supplied model.
    @discardableResult
    public func unload(id: String) async throws -> Bool {
        let publicID = try resolvedPublicID(for: id)
        let modelID = try canonicalModelID(for: publicID)
        let keys = residentKeys(matchingPublicID: publicID, modelID: modelID)
        let loadingKeys = loadingResidentKeys(matchingPublicID: publicID, modelID: modelID)
        var didAcceptUnload = false
        for key in loadingKeys {
            guard var entry = loadingResidents[key], !entry.model.runtime.isPinned else {
                continue
            }
            didAcceptUnload = true
            entry.pendingUnloadAfterLoad = true
            entry.task.cancel()
            loadingResidents[key] = entry
        }
        for key in keys {
            guard var entry = residents[key], !entry.isPinned else {
                continue
            }
            didAcceptUnload = true
            guard entry.leaseCount == 0 else {
                entry.pendingUnloadAfterLease = true
                residents[key] = entry
                continue
            }
            residents.removeValue(forKey: key)
            await entry.session.unload()
        }
        return didAcceptUnload
    }

    /// Requests cancellation of resident sessions for a registered model.
    public func stop(id: String) throws {
        let publicID = try resolvedPublicID(for: id)
        stop(publicID: publicID, modelID: try canonicalModelID(for: publicID))
    }

    /// Requests cancellation of resident sessions matching an explicit model.
    public func stop(model: MLXLanguageModel) {
        stop(publicID: model.model.id, modelID: baseModelID(for: model.model.id))
    }

    /// Evicts idle resident sessions whose per-model TTL has expired.
    public func evictExpired(now: Date = Date()) async {
        let expiredKeys = residents.compactMap { key, entry -> ProviderConfiguration? in
            guard entry.isEvictable, isExpired(entry, now: now) else {
                return nil
            }
            return key
        }
        for key in expiredKeys {
            guard let entry = residents.removeValue(forKey: key) else {
                continue
            }
            await entry.session.unload()
        }
    }

    /// Returns a point-in-time snapshot of pool state.
    public func snapshot() -> Snapshot {
        Snapshot(
            registeredModelIDs: registrations.keys.sorted(),
            aliasTargets: aliasTargets,
            residentModelIDs: residentModelIDs { _ in true },
            pinnedResidentModelIDs: residentModelIDs(where: \.isPinned),
            leasedResidentModelIDs: residentModelIDs { $0.leaseCount > 0 },
            pendingUnloadResidentModelIDs: residentModelIDs(where: \.pendingUnloadAfterLease),
            residentMemoryBytesByModelID: residentMemoryBytesByModelID(),
            servingProfileTargets: servingProfileTargets,
            visibleModels: visibleModels()
        )
    }
}

extension MLXModelPool {
    func visibleModels() -> [MLXModelPoolVisibleModel] {
        let baseModels = registrations.values.map { model in
            visibleModel(
                id: model.model.id,
                sourceModelID: nil,
                aliases: aliases(targeting: model.model.id),
                isServingProfile: false,
                model: model
            )
        }
        let profileModels = servingProfileTargets.compactMap { profileID, modelID in
            visibleServingProfile(id: profileID, modelID: modelID)
        }
        return (baseModels + profileModels).sorted { $0.id < $1.id }
    }

    func visibleServingProfile(
        id profileID: String,
        modelID: String
    ) -> MLXModelPoolVisibleModel? {
        guard let profile = servingProfiles[profileID],
            let model = registrations[modelID]
        else {
            return nil
        }
        return visibleModel(
            id: profileID,
            sourceModelID: modelID,
            aliases: aliases(targeting: profileID),
            isServingProfile: true,
            model: profile.applying(to: model, publicID: profileID)
        )
    }

    func visibleModel(
        id: String,
        sourceModelID: String?,
        aliases: [String],
        isServingProfile: Bool,
        model: MLXLanguageModel
    ) -> MLXModelPoolVisibleModel {
        MLXModelPoolVisibleModel(
            id: id,
            sourceModelID: sourceModelID,
            aliases: aliases,
            isServingProfile: isServingProfile,
            promptStyle: model.model.promptStyle,
            capabilities: model.model.capabilities,
            runtimeKind: runtimeKind(for: model),
            contextLength: model.model.profile?.contextLength,
            maximumResponseTokens: model.maximumResponseTokens
        )
    }

    func aliases(targeting targetID: String) -> [String] {
        aliasTargets.compactMap { alias, target -> String? in
            target == targetID ? alias : nil
        }
        .sorted()
    }

    func runtimeKind(for model: MLXLanguageModel) -> MLXModelRuntimeKind {
        model.model.profile?.runtimeKind ?? (model.model.capabilities.vision ? .vlm : .text)
    }

    func residentSession(
        for model: MLXLanguageModel,
        now: Date
    ) async throws -> any MLXGeneratingSession {
        let key = residentConfiguration(for: model)
        if let entry = residents[key] {
            residents[key]?.lastAccess = now
            return entry.session
        }
        if let loading = loadingResidents[key] {
            return try await finishLoadingResident(for: key, loading: loading, now: now)
        }

        let estimatedResidentBytes = MLXModelPoolMemoryEstimator.estimatedResidentBytes(for: model)
        try await makeRoomForResidentModel(incomingBytes: estimatedResidentBytes)

        let session = sessionFactory()
        let loading = MLXModelPoolLoadingEntry(
            id: UUID(),
            model: model,
            session: session,
            estimatedResidentBytes: estimatedResidentBytes,
            startedAt: now,
            task: Self.makePreloadTask(of: session, configuration: key)
        )
        loadingResidents[key] = loading
        return try await finishLoadingResident(for: key, loading: loading, now: now)
    }

    func leaseSession(
        for model: MLXLanguageModel,
        now: Date
    ) async throws -> any MLXGeneratingSession {
        let key = residentConfiguration(for: model)
        let session = try await residentSession(for: model, now: now)
        residents[key]?.leaseCount += 1
        residents[key]?.lastAccess = now
        return session
    }

    func releaseLease(
        for key: ProviderConfiguration,
        now: Date
    ) async {
        guard var entry = residents[key] else {
            return
        }
        entry.leaseCount = max(0, entry.leaseCount - 1)
        entry.lastAccess = now
        residents[key] = entry
        if entry.leaseCount == 0,
            !entry.isPinned,
            shouldUnloadAfterFinalLease(entry) {
            residents.removeValue(forKey: key)
            await entry.session.unload()
        }
    }

    func stop(
        publicID: String,
        modelID: String
    ) {
        for entry in residents.values where model(
            entry.model,
            matchesPublicID: publicID,
            modelID: modelID
        ) {
            entry.session.stop()
        }
        for entry in loadingResidents.values where model(
            entry.model,
            matchesPublicID: publicID,
            modelID: modelID
        ) {
            entry.session.stop()
        }
    }

    func makeRoomForResidentModel(incomingBytes: Int) async throws {
        while requiresResidentEviction(incomingBytes: incomingBytes) {
            guard let victim = leastRecentlyUsedEvictableKey() else {
                throw residentCapacityError(incomingBytes: incomingBytes)
            }
            guard let entry = residents.removeValue(forKey: victim) else {
                continue
            }
            await entry.session.unload()
        }
    }

    func requiresResidentEviction(incomingBytes: Int) -> Bool {
        if reservedResidentCount >= configuration.maxResidentModels {
            return true
        }
        guard let maxBytes = configuration.maxResidentMemoryBytes else {
            return false
        }
        return reservedResidentMemoryBytes + max(0, incomingBytes) > maxBytes
    }

    func residentCapacityError(incomingBytes: Int) -> MLXModelPoolError {
        if reservedResidentCount >= configuration.maxResidentModels {
            return .capacityExhausted(maxResidentModels: configuration.maxResidentModels)
        }
        return .residentMemoryCapacityExhausted(
            requestedBytes: max(0, incomingBytes),
            maxResidentMemoryBytes: configuration.maxResidentMemoryBytes ?? 0,
            residentBytes: reservedResidentMemoryBytes
        )
    }

    func leastRecentlyUsedEvictableKey() -> ProviderConfiguration? {
        residents
            .filter(\.value.isEvictable)
            .min { lhs, rhs in
                if lhs.value.lastAccess == rhs.value.lastAccess {
                    return lhs.key.modelName < rhs.key.modelName
                }
                return lhs.value.lastAccess < rhs.value.lastAccess
            }?
            .key
    }

    func isExpired(
        _ entry: MLXModelPoolResidentEntry,
        now: Date
    ) -> Bool {
        guard let ttl = entry.model.runtime.idleTTLSeconds else {
            return false
        }
        return now.timeIntervalSince(entry.lastAccess) >= Double(max(0, ttl))
    }

    func shouldUnloadAfterFinalLease(_ entry: MLXModelPoolResidentEntry) -> Bool {
        entry.pendingUnloadAfterLease || entry.model.runtime.residencyPreference == .cold
    }

    func residentModelIDs(
        where shouldInclude: (MLXModelPoolResidentEntry) -> Bool
    ) -> [String] {
        residents.values
            .filter(shouldInclude)
            .map(\.model.model.id)
            .sorted()
    }

    var residentMemoryBytes: Int {
        residents.values.reduce(0) { total, entry in
            total + entry.estimatedResidentBytes
        }
    }

    func residentMemoryBytesByModelID() -> [String: Int] {
        Dictionary(uniqueKeysWithValues: residents.values.map { entry in
            (entry.model.model.id, entry.estimatedResidentBytes)
        })
    }
}
