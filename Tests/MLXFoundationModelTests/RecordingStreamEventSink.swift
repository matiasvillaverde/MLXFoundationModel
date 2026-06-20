@testable import MLXFoundationModel

actor RecordingStreamEventSink: MLXStreamEventSink {
    private typealias EventContinuation = CheckedContinuation<MLXTranslatedStreamEvent, Never>

    private var events: [MLXTranslatedStreamEvent] = []
    private var waiters: [(index: Int, continuation: EventContinuation)] = []

    func send(_ event: MLXTranslatedStreamEvent) {
        events.append(event)
        resumeReadyWaiters()
    }

    func event(at index: Int) async -> MLXTranslatedStreamEvent {
        if events.indices.contains(index) {
            return events[index]
        }

        return await withCheckedContinuation { continuation in
            waiters.append((index: index, continuation: continuation))
        }
    }

    func snapshot() -> [MLXTranslatedStreamEvent] {
        events
    }

    private func resumeReadyWaiters() {
        var pending: [(index: Int, continuation: EventContinuation)] = []
        for waiter in waiters {
            if events.indices.contains(waiter.index) {
                waiter.continuation.resume(returning: events[waiter.index])
            } else {
                pending.append(waiter)
            }
        }
        waiters = pending
    }
}
