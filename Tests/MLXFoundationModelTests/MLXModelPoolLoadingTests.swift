import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX model pool loading")
struct MLXModelPoolLoadingTests {
    @Test("preload failure clears loading reservation and unloads failed session")
    func preloadFailureClearsLoadingReservationAndUnloadsFailedSession() async throws {
        let store = MLXModelPoolRecordingSessionStore(
            preloadFailures: [.preloadFailed, nil]
        )
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 1),
            sessionFactory: store.makeSession
        )
        try await pool.register(Self.model("flaky"))

        await #expect(throws: MLXModelPoolRecordingSessionError.preloadFailed) {
            _ = try await pool.preload(id: "flaky", now: Self.time(0))
        }
        let failedSession = try #require(store.snapshot().first)
        #expect(await failedSession.unloadCount == 1)
        #expect(await pool.snapshot().residentModelIDs.isEmpty)

        _ = try await pool.preload(id: "flaky", now: Self.time(1))

        let sessions = store.snapshot()
        let loadedSession = try #require(sessions.last)
        let snapshot = await pool.snapshot()

        #expect(sessions.count == 2)
        #expect(await loadedSession.preloadCount == 1)
        #expect(await loadedSession.unloadCount == 0)
        #expect(snapshot.residentModelIDs == ["flaky"])
    }

    @Test("unload during preload cancels loading reservation")
    func unloadDuringPreloadCancelsLoadingReservation() async throws {
        let store = MLXModelPoolRecordingSessionStore(preloadDelay: .milliseconds(100))
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        try await pool.register(Self.model("loading"))

        let loadingTask = Task {
            try await pool.preload(id: "loading", now: Self.time(0))
        }
        try await Self.waitForPreloadStart(in: store)

        let didAcceptUnload = try await pool.unload(id: "loading")

        await #expect(throws: CancellationError.self) {
            _ = try await loadingTask.value
        }
        let session = try #require(store.snapshot().first)
        let snapshot = await pool.snapshot()

        #expect(didAcceptUnload)
        #expect(await session.unloadCount == 1)
        #expect(snapshot.residentModelIDs.isEmpty)
    }

    private static func model(_ id: String) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: id,
                location: URL(fileURLWithPath: "/tmp/mlx-model-pool-loading-tests/\(id)")
            )
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
