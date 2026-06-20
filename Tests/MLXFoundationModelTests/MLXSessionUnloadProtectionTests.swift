import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX session unload protection")
struct MLXSessionUnloadProtectionTests {
    @Test("pinned runtime skips unload without pausing admission")
    func pinnedRuntimeSkipsUnloadWithoutPausingAdmission() async throws {
        let session = try await Self.pinnedSession()

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            await session.unload()

            let admission = await session.generationAdmission.snapshot()
            #expect(!admission.admissionPaused)
            #expect(await session.pendingUnloadAfterGeneration == false)
        }

        let snapshots = Self.lifecycleSnapshots(from: recorded.events)
        #expect(snapshots.map(\.stage) == [.unloadSkipped])
        #expect(snapshots.first?.message == "model is pinned")
    }

    @Test("pinned runtime does not defer active-generation unload")
    func pinnedRuntimeDoesNotDeferActiveGenerationUnload() async throws {
        let session = try await Self.pinnedSession()

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            await session.beginGeneration()
            await session.unload()

            let admission = await session.generationAdmission.snapshot()
            #expect(await session.activeGenerationCount == 1)
            #expect(await session.pendingUnloadAfterGeneration == false)
            #expect(!admission.admissionPaused)

            await session.finishGeneration()
        }

        let stages = Self.lifecycleSnapshots(from: recorded.events).map(\.stage)
        #expect(stages.contains(.generationStarted))
        #expect(stages.contains(.unloadSkipped))
        #expect(stages.contains(.generationFinished))
        #expect(!stages.contains(.unloadDeferred))
        #expect(!stages.contains(.unloadAdmissionPaused))
    }

    @Test("deferred unload waits for all active generations")
    func deferredUnloadWaitsForAllActiveGenerations() async throws {
        let session = MLXSession()

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            await session.beginGeneration()
            await session.beginGeneration()
            await session.unload()

            #expect(await session.activeGenerationCount == 2)
            #expect(await session.pendingUnloadAfterGeneration)

            await session.finishGeneration()
            try await Task.sleep(for: .milliseconds(10))
            #expect(await session.activeGenerationCount == 1)
            #expect(await session.pendingUnloadAfterGeneration)

            await session.finishGeneration()
            try await Self.waitUntil {
                let activeGenerationCount = await session.activeGenerationCount
                let pendingUnloadAfterGeneration = await session.pendingUnloadAfterGeneration
                return activeGenerationCount == 0 && !pendingUnloadAfterGeneration
            }
            let admission = await session.generationAdmission.snapshot()
            #expect(!admission.admissionPaused)
        }

        let stages = Self.lifecycleSnapshots(from: recorded.events).map(\.stage)
        #expect(stages.contains(.generationStarted))
        #expect(stages.contains(.generationFinished))
        #expect(stages.contains(.unloadAdmissionPaused))
        #expect(stages.contains(.unloadDeferred))
        #expect(stages.contains(.unloadSkipped))
        #expect(stages.contains(.unloadAdmissionResumed))
    }

    @Test("deferred unload holds queued admissions until unload finishes")
    func deferredUnloadHoldsQueuedAdmissionsUntilUnloadFinishes() async throws {
        let session = MLXSession()
        let activeLease = try await session.generationAdmission.acquire()
        await session.beginGeneration()

        await session.unload()
        let queuedTask = Task {
            try await session.generationAdmission.acquire()
        }
        try await Task.sleep(for: .milliseconds(10))

        let paused = await session.generationAdmission.snapshot()
        #expect(paused.admissionPaused)
        #expect(paused.waitingCount == 1)

        await session.finishGeneration()
        await session.generationAdmission.release(activeLease)

        try await Self.waitUntil {
            let snapshot = await session.generationAdmission.snapshot()
            return !snapshot.admissionPaused && snapshot.activeCount == 1
        }
        let queuedLease = try await Self.value(from: queuedTask)
        await session.generationAdmission.release(queuedLease)
    }

    private static func waitUntil(
        condition: () async -> Bool
    ) async throws {
        for _ in 0 ..< 100 {
            if await condition() {
                return
            }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw TimeoutError()
    }

    private static func value<T: Sendable>(
        from task: Task<T, any Error>
    ) async throws -> T {
        try await withThrowingTaskGroup(of: T.self) { group in
            group.addTask {
                try await task.value
            }
            group.addTask {
                try await Task.sleep(for: .seconds(1))
                throw TimeoutError()
            }

            guard let value = try await group.next() else {
                throw TimeoutError()
            }
            group.cancelAll()
            return value
        }
    }

    private static func lifecycleSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXSessionLifecycleSnapshot] {
        events.compactMap { event in
            guard case .sessionLifecycle(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func pinnedSession() async throws -> MLXSession {
        let session = MLXSession(configuration: ProviderConfiguration(
            location: URL(fileURLWithPath: "/tmp/pinned-mlx-model"),
            modelName: "pinned-test-model",
            runtime: ModelRuntimePreferences(isPinned: true)
        ))
        try await session.applyRuntimeConfiguration()
        return session
    }

    private struct TimeoutError: Error {}
}
