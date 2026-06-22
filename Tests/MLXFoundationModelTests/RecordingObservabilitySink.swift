import Foundation
import MLXLocalModels

final class RecordingObservabilitySink: MLXObservabilitySink, @unchecked Sendable {
    private let lock = NSLock()
    private var recordedEvents: [MLXObservabilityEvent] = []
    private var recordedRequests: [MLXRequestSummary] = []

    deinit {
        lock.lock()
        recordedEvents.removeAll(keepingCapacity: false)
        recordedRequests.removeAll(keepingCapacity: false)
        lock.unlock()
    }

    func record(_ event: MLXObservabilityEvent) {
        lock.lock()
        recordedEvents.append(event)
        lock.unlock()
    }

    func recordRequest(_ summary: MLXRequestSummary) {
        lock.lock()
        recordedRequests.append(summary)
        lock.unlock()
    }

    func events() -> [MLXObservabilityEvent] {
        lock.lock()
        let snapshot = recordedEvents
        lock.unlock()
        return snapshot
    }

    func requests() -> [MLXRequestSummary] {
        lock.lock()
        let snapshot = recordedRequests
        lock.unlock()
        return snapshot
    }
}
