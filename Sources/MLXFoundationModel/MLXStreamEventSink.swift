protocol MLXStreamEventSink: Sendable {
    func send(_ event: MLXTranslatedStreamEvent) async
}

extension MLXStreamEventSink {
    func finish() async {
        // Most sinks do not retain end-of-stream state.
    }
}
