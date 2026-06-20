import Foundation
@testable import MLXFoundationModel
import MLXLocalModels

actor MLXModelPoolRecordingSession: MLXGeneratingSession {
    private let preloadDelay: Duration?
    private let preloadFailure: MLXModelPoolRecordingSessionError?
    private(set) var preloadConfigurations: [ProviderConfiguration] = []
    private(set) var streamInputs: [LLMInput] = []
    private(set) var unloadCount = 0

    init(
        preloadDelay: Duration? = nil,
        preloadFailure: MLXModelPoolRecordingSessionError? = nil
    ) {
        self.preloadDelay = preloadDelay
        self.preloadFailure = preloadFailure
    }

    func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, any Error> {
        streamInputs.append(input)
        return AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamChunk(text: "recorded", event: .text, tokenCount: 1))
            continuation.finish()
        }
    }

    nonisolated func stop() {
        // Test fixture has no active generation loop to cancel.
    }

    func preload(configuration: ProviderConfiguration) -> AsyncThrowingStream<Progress, any Error> {
        preloadConfigurations.append(configuration)
        return AsyncThrowingStream { continuation in
            let delay = preloadDelay
            let failure = preloadFailure
            Task<Void, Never> {
                if let delay {
                    try? await Task.sleep(for: delay)
                }
                if let failure {
                    continuation.finish(throwing: failure)
                } else {
                    continuation.finish()
                }
            }
        }
    }

    func unload() async {
        unloadCount += 1
    }

    var preloadCount: Int {
        preloadConfigurations.count
    }

    var streamCallCount: Int {
        streamInputs.count
    }
}
