#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import MLXLocalModels

actor MLXPrewarmCoordinator {
    private struct Operation {
        let id: UUID
        let configuration: ProviderConfiguration
        let task: Task<Void, any Error>
    }

    private var loadedConfiguration: ProviderConfiguration?
    private var inFlightOperation: Operation?

    nonisolated func prewarm(
        configuration: ProviderConfiguration,
        session: any MLXGeneratingSession
    ) {
        Task {
            await startPreload(configuration: configuration, session: session)
        }
    }

    func ensureLoaded(
        configuration: ProviderConfiguration,
        session: any MLXGeneratingSession
    ) async throws {
        if loadedConfiguration == configuration {
            return
        }

        let operation = currentOrStartedOperation(
            configuration: configuration,
            session: session
        )
        try await operation.task.value
    }

    private func startPreload(
        configuration: ProviderConfiguration,
        session: any MLXGeneratingSession
    ) {
        guard loadedConfiguration != configuration,
            inFlightOperation?.configuration != configuration
        else {
            return
        }
        inFlightOperation?.task.cancel()
        _ = currentOrStartedOperation(configuration: configuration, session: session)
    }

    private func currentOrStartedOperation(
        configuration: ProviderConfiguration,
        session: any MLXGeneratingSession
    ) -> Operation {
        if let operation = inFlightOperation,
            operation.configuration == configuration {
            return operation
        }

        let operationID = UUID()
        let task = Self.makePreloadTask(configuration: configuration, session: session)
        let operation = Operation(
            id: operationID,
            configuration: configuration,
            task: task
        )
        inFlightOperation = operation
        Task {
            do {
                try await task.value
                completePreload(id: operationID, configuration: configuration)
            } catch {
                failPreload(id: operationID)
            }
        }
        return operation
    }

    private func completePreload(
        id: UUID,
        configuration: ProviderConfiguration
    ) {
        guard inFlightOperation?.id == id else {
            return
        }
        loadedConfiguration = configuration
        inFlightOperation = nil
    }

    private func failPreload(id: UUID) {
        guard inFlightOperation?.id == id else {
            return
        }
        inFlightOperation = nil
    }

    private static func makePreloadTask(
        configuration: ProviderConfiguration,
        session: any MLXGeneratingSession
    ) -> Task<Void, any Error> {
        Task {
            let progress = await session.preload(configuration: configuration)
            for try await _ in progress {
                // Drain preload progress before the model is considered warm.
            }
        }
    }
}
#endif
