import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX pooled session")
struct MLXPooledSessionTests {
    @Test("streams through pool and releases cold residency")
    func streamsThroughPoolAndReleasesColdResidency() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        let model = Self.model(
            "cold",
            runtime: ModelRuntimePreferences(residencyPreference: .cold)
        )
        let session = MLXPooledSession(model: model, pool: pool)

        let stream = await session.stream(LLMInput(context: "Hello"))
        var chunks: [LLMStreamChunk] = []
        for try await chunk in stream {
            chunks.append(chunk)
        }

        let recording = try #require(store.snapshot().first)
        let snapshot = await pool.snapshot()

        #expect(chunks.map(\.text) == ["recorded"])
        #expect(await recording.preloadCount == 1)
        #expect(await recording.streamCallCount == 1)
        #expect(await recording.unloadCount == 1)
        #expect(snapshot.residentModelIDs.isEmpty)
    }

    @Test("preloads through shared pool")
    func preloadsThroughSharedPool() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        let model = Self.model("warm")
        let session = MLXPooledSession(model: model, pool: pool)

        let progress = await session.preload(configuration: model.providerConfiguration)
        var completedUnits: [Int64] = []
        for try await value in progress {
            completedUnits.append(value.completedUnitCount)
        }

        let recording = try #require(store.snapshot().first)
        let snapshot = await pool.snapshot()

        #expect(completedUnits == [100])
        #expect(await recording.preloadCount == 1)
        #expect(snapshot.residentModelIDs == ["warm"])
    }

    private static func model(
        _ id: String,
        runtime: ModelRuntimePreferences = .default
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: id,
                location: URL(fileURLWithPath: "/tmp/mlx-pooled-session-tests/\(id)")
            ),
            runtime: runtime
        )
    }
}
