@testable import MLXFoundationModel
import Testing

@Suite("MLX Harmony stream filter")
struct MLXHarmonyStreamFilterTests {
    @Test("streams analysis as think block while suppressing markers")
    func streamsAnalysisAsThinkBlockWhileSuppressingMarkers() {
        var filter = MLXHarmonyStreamFilter()
        let chunks = [
            filter.feed("<|chan"),
            filter.feed("nel|>analysis<|message|>thinking<|end|>"),
            filter.feed("<|start|>assistant<|channel|>final<|message|>Answer"),
            filter.feed("<|return|>"),
            filter.finish()
        ]

        #expect(chunks == ["", "<think>\nthinking</think>\n", "Answer", "", ""])
        #expect(chunks.joined() == "<think>\nthinking</think>\nAnswer")
    }

    @Test("suppresses commentary tool calls")
    func suppressesCommentaryToolCalls() {
        var filter = MLXHarmonyStreamFilter()
        let output = [
            filter.feed(" to=functions.weather<|channel|>commentary<|message|>"),
            filter.feed(#"{"city":"Berlin"}"#),
            filter.feed("<|call|><|start|>assistant<|channel|>final<|message|>Done"),
            filter.finish()
        ].joined()

        #expect(output == "Done")
    }

    @Test("falls back to visible text for unknown channels without recipient")
    func fallsBackToVisibleTextForUnknownChannelsWithoutRecipient() {
        var filter = MLXHarmonyStreamFilter()
        let output = [
            filter.feed("<|channel|>mardown<|message|>visible"),
            filter.feed(" text<|return|>"),
            filter.finish()
        ].joined()

        #expect(output == "visible text")
    }

    @Test("preserves non Harmony text")
    func preservesNonHarmonyText() {
        var filter = MLXHarmonyStreamFilter()
        let output = [
            filter.feed("plain answer"),
            filter.finish()
        ].joined()

        #expect(output == "plain answer")
    }
}
