@testable import MLXFoundationModel
import Testing

@Suite("MLX think tag stream splitter")
struct MLXThinkTagStreamSplitterTests {
    @Test("splits complete think block into reasoning and response segments")
    func splitsCompleteThinkBlock() {
        var splitter = MLXThinkTagStreamSplitter()
        let segments = splitter.consume("<think>\nreasoning</think>\nAnswer") + splitter.finish()

        #expect(segments == [
            .init(kind: .reasoning, text: "reasoning"),
            .init(kind: .response, text: "Answer")
        ])
    }

    @Test("waits for split markers before emitting text")
    func waitsForSplitMarkers() {
        var splitter = MLXThinkTagStreamSplitter()
        var segments: [MLXThinkTagStreamSplitter.Segment] = []

        segments.append(contentsOf: splitter.consume("<thi"))
        segments.append(contentsOf: splitter.consume("nk>hidden</thi"))
        segments.append(contentsOf: splitter.consume("nk>shown"))
        segments.append(contentsOf: splitter.finish())

        #expect(segments == [
            .init(kind: .reasoning, text: "hidden"),
            .init(kind: .response, text: "shown")
        ])
    }

    @Test("preserves ordinary text and unfinished marker prose")
    func preservesOrdinaryTextAndUnfinishedMarkerProse() {
        var splitter = MLXThinkTagStreamSplitter()
        let segments = splitter.consume("Use <think as plain text") + splitter.finish()

        #expect(segments == [
            .init(kind: .response, text: "Use <think as plain text")
        ])
    }

    @Test("recovers unclosed thinking as response when no answer was emitted")
    func recoversUnclosedThinkingAsResponseWhenNoAnswerWasEmitted() {
        var splitter = MLXThinkTagStreamSplitter()
        let segments = splitter.consume("<think>\nbody without close") + splitter.finish()

        #expect(segments == [
            .init(kind: .reasoning, text: "body without close"),
            .init(kind: .response, text: "body without close")
        ])
    }

    @Test("does not recover unclosed thinking when visible content already streamed")
    func doesNotRecoverUnclosedThinkingWhenVisibleContentAlreadyStreamed() {
        var splitter = MLXThinkTagStreamSplitter()
        let segments = splitter.consume("visible <think>\nprivate") + splitter.finish()

        #expect(segments == [
            .init(kind: .response, text: "visible "),
            .init(kind: .reasoning, text: "private")
        ])
    }

    @Test("starts in reasoning when prompt already opened thinking")
    func startsInReasoningWhenPromptAlreadyOpenedThinking() {
        var splitter = MLXThinkTagStreamSplitter(startInReasoning: true)
        let segments = splitter.consume("reasoning</think>\nAnswer") + splitter.finish()

        #expect(segments == [
            .init(kind: .reasoning, text: "reasoning"),
            .init(kind: .response, text: "Answer")
        ])
    }

    @Test("buffers split close marker while starting in reasoning")
    func buffersSplitCloseMarkerWhileStartingInReasoning() {
        var splitter = MLXThinkTagStreamSplitter(startInReasoning: true)
        var segments: [MLXThinkTagStreamSplitter.Segment] = []

        segments.append(contentsOf: splitter.consume("reas</thi"))
        segments.append(contentsOf: splitter.consume("nk>shown"))
        segments.append(contentsOf: splitter.finish())

        #expect(segments == [
            .init(kind: .reasoning, text: "reas"),
            .init(kind: .response, text: "shown")
        ])
    }

    @Test("drops duplicate open marker when prompt already opened thinking")
    func dropsDuplicateOpenMarkerWhenPromptAlreadyOpenedThinking() {
        var splitter = MLXThinkTagStreamSplitter(startInReasoning: true)
        let segments = splitter.consume("<think>\nreasoning</think>Answer") + splitter.finish()

        #expect(segments == [
            .init(kind: .reasoning, text: "reasoning"),
            .init(kind: .response, text: "Answer")
        ])
    }
}
