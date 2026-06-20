import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX model pool")
struct MLXModelPoolTests {
    @Test("reuses resident session through aliases")
    func reusesResidentSessionThroughAliases() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model("qwen"), aliases: ["default"])

        _ = try await pool.preload(id: "default", now: Self.time(0))
        _ = try await pool.preload(id: "qwen", now: Self.time(1))

        let sessions = store.snapshot()
        let first = try #require(sessions.first)
        let snapshot = await pool.snapshot()

        #expect(sessions.count == 1)
        #expect(await first.preloadCount == 1)
        #expect(snapshot.residentModelIDs == ["qwen"])
        #expect(snapshot.aliasTargets == ["default": "qwen"])
    }

    @Test("resolves serving profiles as API-visible model variants")
    func resolvesServingProfilesAsAPIVisibleModelVariants() async throws {
        let pool = MLXModelPool()
        try await pool.register(
            Self.model("qwen", maximumResponseTokens: 512),
            aliases: ["default"],
            profiles: [
                MLXModelServingProfile(
                    name: "deterministic",
                    aliases: ["stable"],
                    sampling: .deterministic,
                    maximumResponseTokens: 128
                )
            ]
        )

        let model = try await pool.model(id: "stable")
        let snapshot = await pool.snapshot()

        #expect(model.model.id == "qwen:deterministic")
        #expect(model.model.location.path.hasSuffix("/qwen"))
        #expect(model.sampling == .deterministic)
        #expect(model.maximumResponseTokens == 128)
        #expect(snapshot.servingProfileTargets == ["qwen:deterministic": "qwen"])
        #expect(snapshot.aliasTargets["stable"] == "qwen:deterministic")
        #expect(snapshot.aliasTargets["default"] == "qwen")

        let visibleProfile = try #require(snapshot.visibleModels.first { model in
            model.id == "qwen:deterministic"
        })
        #expect(visibleProfile.sourceModelID == "qwen")
        #expect(visibleProfile.aliases == ["stable"])
        #expect(visibleProfile.isServingProfile)
        #expect(visibleProfile.maximumResponseTokens == 128)
    }

    @Test("sampling-only serving profiles reuse resident model weights")
    func samplingOnlyServingProfilesReuseResidentModelWeights() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(
            Self.model("qwen"),
            profiles: [
                MLXModelServingProfile(
                    name: "fast",
                    sampling: SamplingParameters(
                        temperature: 0,
                        topP: 1,
                        topK: 1
                    )
                )
            ]
        )

        _ = try await pool.preload(id: "qwen", now: Self.time(0))
        _ = try await pool.preload(id: "qwen:fast", now: Self.time(1))

        let sessions = store.snapshot()
        let first = try #require(sessions.first)
        let snapshot = await pool.snapshot()
        let preloadConfigurations = await first.preloadConfigurations

        #expect(sessions.count == 1)
        #expect(await first.preloadCount == 1)
        #expect(preloadConfigurations.map(\.modelName) == ["qwen"])
        #expect(snapshot.residentModelIDs == ["qwen"])
    }

    @Test("concurrent same-model preloads share one loading reservation")
    func concurrentSameModelPreloadsShareOneLoadingReservation() async throws {
        let store = MLXModelPoolRecordingSessionStore(preloadDelay: .milliseconds(50))
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model("shared"))

        async let first = pool.preload(id: "shared", now: Self.time(0))
        async let second = pool.preload(id: "shared", now: Self.time(1))

        _ = try await first
        _ = try await second

        let sessions = store.snapshot()
        let session = try #require(sessions.first)
        let snapshot = await pool.snapshot()

        #expect(sessions.count == 1)
        #expect(await session.preloadCount == 1)
        #expect(snapshot.residentModelIDs == ["shared"])
    }

    @Test("loading reservations count against resident capacity")
    func loadingReservationsCountAgainstResidentCapacity() async throws {
        let store = MLXModelPoolRecordingSessionStore(preloadDelay: .milliseconds(100))
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 1),
            sessionFactory: store.makeSession
        )
        try await pool.register(Self.model("loading"))
        try await pool.register(Self.model("other"))

        async let loading = pool.preload(id: "loading", now: Self.time(0))
        try await Self.waitForPreloadStart(in: store)

        await #expect(throws: MLXModelPoolError.capacityExhausted(maxResidentModels: 1)) {
            _ = try await pool.preload(id: "other", now: Self.time(1))
        }
        _ = try await loading

        let sessions = store.snapshot()
        let snapshot = await pool.snapshot()

        #expect(sessions.count == 1)
        #expect(snapshot.residentModelIDs == ["loading"])
    }

    @Test("evicts least recently used unpinned resident")
    func evictsLeastRecentlyUsedUnpinnedResident() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 2),
            sessionFactory: store.makeSession
        )
        try await pool.register(Self.model("a"))
        try await pool.register(Self.model("b"))
        try await pool.register(Self.model("c"))

        _ = try await pool.preload(id: "a", now: Self.time(0))
        _ = try await pool.preload(id: "b", now: Self.time(1))
        _ = try await pool.preload(id: "a", now: Self.time(2))
        _ = try await pool.preload(id: "c", now: Self.time(3))

        let snapshot = await pool.snapshot()
        let sessions = store.snapshot()

        #expect(snapshot.residentModelIDs == ["a", "c"])
        #expect(await sessions[1].unloadCount == 1)
        #expect(await sessions[0].unloadCount == 0)
        #expect(await sessions[2].unloadCount == 0)
    }

    @Test("pinned resident blocks capacity eviction")
    func pinnedResidentBlocksCapacityEviction() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model(
            "pinned",
            runtime: ModelRuntimePreferences(isPinned: true)
        ))
        try await pool.register(Self.model("other"))

        _ = try await pool.preload(id: "pinned", now: Self.time(0))

        await #expect(throws: MLXModelPoolError.capacityExhausted(maxResidentModels: 1)) {
            _ = try await pool.preload(id: "other", now: Self.time(1))
        }
        let snapshot = await pool.snapshot()

        #expect(snapshot.residentModelIDs == ["pinned"])
        #expect(snapshot.pinnedResidentModelIDs == ["pinned"])
    }

    @Test("idle TTL evicts expired unpinned residents")
    func idleTTLEvictsExpiredUnpinnedResidents() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 2),
            sessionFactory: store.makeSession
        )
        try await pool.register(Self.model(
            "short",
            runtime: ModelRuntimePreferences(idleTTLSeconds: 10)
        ))
        try await pool.register(Self.model(
            "pinned",
            runtime: ModelRuntimePreferences(isPinned: true, idleTTLSeconds: 10)
        ))

        _ = try await pool.preload(id: "short", now: Self.time(0))
        _ = try await pool.preload(id: "pinned", now: Self.time(0))
        await pool.evictExpired(now: Self.time(11))

        let snapshot = await pool.snapshot()
        let sessions = store.snapshot()

        #expect(snapshot.residentModelIDs == ["pinned"])
        #expect(await sessions[0].unloadCount == 1)
        #expect(await sessions[1].unloadCount == 0)
    }

    @Test("cold residency unloads after scoped use")
    func coldResidencyUnloadsAfterScopedUse() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model(
            "cold",
            runtime: ModelRuntimePreferences(residencyPreference: .cold)
        ))

        let modelID = try await pool.withSession(id: "cold", now: Self.time(0)) { _ in
            "cold"
        }
        let snapshot = await pool.snapshot()
        let session = try #require(store.snapshot().first)

        #expect(modelID == "cold")
        #expect(snapshot.residentModelIDs.isEmpty)
        #expect(await session.unloadCount == 1)
    }

    @Test("explicit unload waits for leased resident then unloads")
    func explicitUnloadWaitsForLeasedResidentThenUnloads() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model("warm"))

        let modelID = try await pool.withSession(id: "warm", now: Self.time(0)) { _ in
            let didAcceptUnload = try await pool.unload(id: "warm")
            let snapshot = await pool.snapshot()
            let session = try #require(store.snapshot().first)

            #expect(didAcceptUnload)
            #expect(snapshot.residentModelIDs == ["warm"])
            #expect(snapshot.leasedResidentModelIDs == ["warm"])
            #expect(snapshot.pendingUnloadResidentModelIDs == ["warm"])
            #expect(await session.unloadCount == 0)
            return "warm"
        }
        let snapshot = await pool.snapshot()
        let session = try #require(store.snapshot().first)

        #expect(modelID == "warm")
        #expect(snapshot.residentModelIDs.isEmpty)
        #expect(snapshot.pendingUnloadResidentModelIDs.isEmpty)
        #expect(await session.unloadCount == 1)
    }

    private static func model(
        _ id: String,
        runtime: ModelRuntimePreferences = .default,
        sampling: SamplingParameters = .default,
        maximumResponseTokens: Int = 2_048
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: id,
                location: URL(fileURLWithPath: "/tmp/mlx-model-pool-tests/\(id)")
            ),
            runtime: runtime,
            sampling: sampling,
            maximumResponseTokens: maximumResponseTokens
        )
    }

    private static func time(_ seconds: TimeInterval) -> Date {
        Date(timeIntervalSince1970: seconds)
    }

    private static func waitForPreloadStart(
        in store: MLXModelPoolRecordingSessionStore
    ) async throws {
        for _ in 0..<100 {
            if let session = store.snapshot().first,
                await session.preloadCount > 0 {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        Issue.record("Timed out waiting for recording preload to start")
    }
}
