@testable import MLXFoundationModel

actor MLXRealModelRecordingStreamEventSink: MLXStreamEventSink {
    private var events: [MLXTranslatedStreamEvent] = []

    func send(_ event: MLXTranslatedStreamEvent) {
        events.append(event)
    }

    func snapshot() -> [MLXTranslatedStreamEvent] {
        events
    }
}
