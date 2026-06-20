import Foundation
import MLXLocalModels

extension MLXModelPool {
    func finishLoadingResident(
        for key: ProviderConfiguration,
        loading: MLXModelPoolLoadingEntry,
        now: Date
    ) async throws -> any MLXGeneratingSession {
        do {
            try await loading.task.value
        } catch {
            if let failed = removeLoadingResident(for: key, matching: loading) {
                await failed.session.unload()
            }
            throw error
        }

        if let entry = residents[key] {
            residents[key]?.lastAccess = now
            _ = removeLoadingResident(for: key, matching: loading)
            return entry.session
        }

        guard let loaded = removeLoadingResident(for: key, matching: loading) else {
            throw CancellationError()
        }
        if loaded.pendingUnloadAfterLoad {
            await loaded.session.unload()
            throw CancellationError()
        }
        residents[key] = MLXModelPoolResidentEntry(
            model: loaded.model,
            session: loaded.session,
            estimatedResidentBytes: loaded.estimatedResidentBytes,
            lastAccess: now
        )
        return loaded.session
    }

    func removeLoadingResident(
        for key: ProviderConfiguration,
        matching loading: MLXModelPoolLoadingEntry
    ) -> MLXModelPoolLoadingEntry? {
        guard loadingResidents[key]?.id == loading.id else {
            return nil
        }
        return loadingResidents.removeValue(forKey: key)
    }

    var reservedResidentCount: Int {
        residents.count + loadingResidents.count
    }

    var reservedResidentMemoryBytes: Int {
        residentMemoryBytes + loadingResidents.values.reduce(0) { total, entry in
            total + entry.estimatedResidentBytes
        }
    }

    static func drainPreload(
        of session: any MLXGeneratingSession,
        configuration: ProviderConfiguration
    ) async throws {
        let progress = await session.preload(configuration: configuration)
        for try await _ in progress {
            // Drain progress so preloading completes before the session is marked resident.
        }
    }

    static func makePreloadTask(
        of session: any MLXGeneratingSession,
        configuration: ProviderConfiguration
    ) -> Task<Void, any Error> {
        Task {
            try await drainPreload(of: session, configuration: configuration)
        }
    }
}
