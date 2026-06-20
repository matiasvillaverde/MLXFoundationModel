import Foundation
import MLXLocalModels

/// ``MLXGeneratingSession`` implementation backed by an ``MLXModelPool``.
///
/// Use this when a host wants FoundationModels-style sessions while sharing
/// resident MLX model instances across executors or requests.
public actor MLXPooledSession: MLXGeneratingSession {
    private let model: MLXLanguageModel
    private let pool: MLXModelPool

    /// Creates a pooled session for a specific model.
    public init(
        model: MLXLanguageModel,
        pool: MLXModelPool
    ) {
        self.model = model
        self.pool = pool
    }

    /// Streams generation through the model pool.
    public func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        pool.stream(for: model, input: input)
    }

    /// Stops the currently resident model session, if one exists.
    nonisolated public func stop() {
        Task {
            await pool.stop(model: model)
        }
    }

    /// Preloads the model through the pool.
    public func preload(
        configuration: ProviderConfiguration
    ) -> AsyncThrowingStream<Progress, any Error> {
        let model = model(overridingWith: configuration)
        let pool = pool
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    _ = try await pool.session(for: model)
                    let progress = Progress(totalUnitCount: 100)
                    progress.completedUnitCount = 100
                    continuation.yield(progress)
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

    /// Unloads idle, unpinned resident sessions for this model.
    public func unload() async {
        _ = try? await pool.unload(id: model.model.id)
    }

    private func model(
        overridingWith configuration: ProviderConfiguration
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: configuration.modelName,
                location: configuration.location,
                promptStyle: model.model.promptStyle,
                capabilities: model.model.capabilities,
                profile: model.model.profile
            ),
            compute: configuration.compute,
            runtime: configuration.runtime ?? model.runtime,
            sampling: model.sampling,
            maximumResponseTokens: model.maximumResponseTokens
        )
    }
}
