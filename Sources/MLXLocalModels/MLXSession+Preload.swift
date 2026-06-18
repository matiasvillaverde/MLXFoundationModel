import Foundation

extension MLXSession {
    /// Preload a model into memory with progress streaming.
    internal func preload(configuration: ProviderConfiguration) -> AsyncThrowingStream<Progress, Error> {
        AsyncThrowingStream { continuation in
            let stopFlag = self.stopFlag
            let preloadTask = Task {
                await self.performPreload(
                    configuration: configuration,
                    stopFlag: stopFlag,
                    continuation: continuation
                )
            }
            continuation.onTermination = { @Sendable termination in
                guard case .cancelled = termination else {
                    return
                }
                stopFlag.set(true)
                preloadTask.cancel()
            }
        }
    }

    private func canReuseLoadedModel(for newConfiguration: ProviderConfiguration) -> Bool {
        guard modelContainer != nil, let configuration else {
            return false
        }
        return configuration.location.standardizedFileURL == newConfiguration.location.standardizedFileURL
            && configuration.modelName == newConfiguration.modelName
    }

    private func performPreload(
        configuration: ProviderConfiguration,
        stopFlag: StopFlag,
        continuation: AsyncThrowingStream<Progress, Error>.Continuation
    ) async {
        do {
            let canReuseLoadedModel = canReuseLoadedModel(for: configuration)
            self.configuration = configuration
            if canReuseLoadedModel {
                try await applyRuntimeConfiguration()
                let progress = Progress(totalUnitCount: 100)
                progress.completedUnitCount = 100
                continuation.yield(progress)
                continuation.finish()
                return
            }
            if modelContainer != nil {
                logger.info("Switching MLX model to \(configuration.location.path)")
                try? await persistPromptCacheIfNeeded()
                modelContainer = nil
            }
            stopFlag.reset()
            try Task.checkCancellation()
            try await streamModelLoad(continuation: continuation)
            try Task.checkCancellation()
            if stopFlag.get() {
                throw CancellationError()
            }
            continuation.finish()
        } catch is CancellationError {
            continuation.finish(throwing: CancellationError())
        } catch {
            continuation.finish(throwing: error)
        }
    }
}
