@testable import MLXFoundationModel
import Testing

@Suite("MLX MiniMax M3 stream filter")
struct MLXMiniMaxM3StreamFilterTests {
    @Test("normalizes split thinking markers")
    func normalizesSplitThinkingMarkers() {
        var filter = MLXMiniMaxM3StreamFilter()

        let output = [
            filter.feed("<mm:thi"),
            filter.feed("nk>reasoning</mm:"),
            filter.feed("think>Answer"),
            filter.finish()
        ].joined()

        #expect(output == "<think>reasoning</think>Answer")
    }

    @Test("drops split MiniMax M3 special tokens")
    func dropsSplitMiniMaxM3SpecialTokens() {
        var filter = MLXMiniMaxM3StreamFilter()

        let output = [
            filter.feed("before [e"),
            filter.feed("~[ ]~"),
            filter.feed("b]]~!"),
            filter.feed("b[]!p"),
            filter.feed("~[]!d~[ after"),
            filter.finish()
        ].joined()

        #expect(output == "before   after")
    }

    @Test("preserves normal text across chunks")
    func preservesNormalTextAcrossChunks() {
        var filter = MLXMiniMaxM3StreamFilter()

        let output = [
            filter.feed("This is "),
            filter.feed("visible."),
            filter.finish()
        ].joined()

        #expect(output == "This is visible.")
    }
}
