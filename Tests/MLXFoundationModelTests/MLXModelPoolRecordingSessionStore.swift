import Foundation
@testable import MLXFoundationModel
import MLXLocalModels

final class MLXModelPoolRecordingSessionStore: @unchecked Sendable {
    private let lock = NSLock()
    private let preloadDelay: Duration?
    private let preloadFailure: MLXModelPoolRecordingSessionError?
    private let preloadFailures: [MLXModelPoolRecordingSessionError?]
    private var sessions: [MLXModelPoolRecordingSession] = []
    private var nextSessionIndex = 0

    init(
        preloadDelay: Duration? = nil,
        preloadFailure: MLXModelPoolRecordingSessionError? = nil,
        preloadFailures: [MLXModelPoolRecordingSessionError?] = []
    ) {
        self.preloadDelay = preloadDelay
        self.preloadFailure = preloadFailure
        self.preloadFailures = preloadFailures
    }

    func makeSession() -> any MLXGeneratingSession {
        let failure = nextPreloadFailure()
        let session = MLXModelPoolRecordingSession(
            preloadDelay: preloadDelay,
            preloadFailure: failure
        )
        lock.withLock {
            sessions.append(session)
        }
        return session
    }

    func nextPreloadFailure() -> MLXModelPoolRecordingSessionError? {
        lock.withLock {
            defer { nextSessionIndex += 1 }
            guard nextSessionIndex < preloadFailures.count else {
                return preloadFailure
            }
            return preloadFailures[nextSessionIndex]
        }
    }

    func snapshot() -> [MLXModelPoolRecordingSession] {
        lock.withLock {
            sessions
        }
    }

    deinit {
        // Required by the repository lint profile for classes.
    }
}
