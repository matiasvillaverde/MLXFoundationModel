#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
import Testing

@Suite("MLX executor prewarm")
struct MLXExecutorPrewarmTests {
    @Test("prewarm starts model preload")
    func prewarmStartsModelPreload() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let model = Self.model()
        let session = RecordingPreloadSession()
        let executor = try MLXExecutor(
            configuration: Self.executorConfiguration(for: model),
            session: session
        )

        executor.prewarm(model: model, transcript: Self.transcript(prompt: "Warm up."))

        try await Self.waitUntil {
            await session.preloadCallCount() == 1
        }
    }

    @Test("coalesces duplicate prewarm requests")
    func coalescesDuplicatePrewarmRequests() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let model = Self.model()
        let session = RecordingPreloadSession()
        let executor = try MLXExecutor(
            configuration: Self.executorConfiguration(for: model),
            session: session
        )

        executor.prewarm(model: model, transcript: Self.transcript(prompt: "Warm up."))
        executor.prewarm(model: model, transcript: Self.transcript(prompt: "Warm up."))

        try await Self.waitUntil {
            await session.preloadCallCount() == 1
        }
    }

    @Test("pooled executor prewarms through shared pool")
    func pooledExecutorPrewarmsThroughSharedPool() async throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let model = Self.model()
        let store = MLXModelPoolRecordingSessionStore()
        let pool = MLXModelPool(sessionFactory: store.makeSession)
        let executor = try MLXExecutor(
            configuration: Self.executorConfiguration(for: model),
            pool: pool
        )

        executor.prewarm(model: model, transcript: Self.transcript(prompt: "Warm up."))

        try await Self.waitUntil {
            guard let session = store.snapshot().first else {
                return false
            }
            return await session.preloadCount == 1
        }
        #expect(await pool.snapshot().residentModelIDs == ["prewarm-fixture"])
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func model() -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: "prewarm-fixture",
                location: URL(fileURLWithPath: "/tmp/MLXFoundationModelPrewarm"),
                promptStyle: .chatML,
                capabilities: MLXModelCapabilities(toolCalling: true, structuredOutput: true)
            )
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func executorConfiguration(
        for model: MLXLanguageModel
    ) -> MLXExecutor.Configuration {
        MLXExecutor.Configuration(
            model: model.model,
            compute: model.compute,
            runtime: model.runtime,
            sampling: model.sampling,
            maximumResponseTokens: model.maximumResponseTokens
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func transcript(prompt: String) -> Transcript {
        Transcript(entries: [
            .prompt(.init(segments: [.text(.init(content: prompt))]))
        ])
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
        Issue.record("Timed out waiting for async condition")
    }

    private actor RecordingPreloadSession: MLXGeneratingSession {
        private var preloadConfigurations: [ProviderConfiguration] = []

        func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
            _ = input
            return AsyncThrowingStream { continuation in
                continuation.finish()
            }
        }

        nonisolated func stop() {
            // No active generation in contract tests.
        }

        func preload(
            configuration: ProviderConfiguration
        ) -> AsyncThrowingStream<Progress, any Error> {
            preloadConfigurations.append(configuration)
            return AsyncThrowingStream { continuation in
                let progress = Progress(totalUnitCount: 100)
                progress.completedUnitCount = 100
                continuation.yield(progress)
                continuation.finish()
            }
        }

        func unload() async {
            // Nothing to unload in the recording session.
        }

        func preloadCallCount() -> Int {
            preloadConfigurations.count
        }
    }
}
#endif
