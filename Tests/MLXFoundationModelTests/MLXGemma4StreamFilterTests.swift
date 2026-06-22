@testable import MLXFoundationModel
import Testing

@Suite("MLX Gemma 4 stream filter")
struct MLXGemma4StreamFilterTests {
    @Test("rewrites thought channel to think tags")
    func rewritesThoughtChannelToThinkTags() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("<|chan"),
            filter.feed("nel>thought\nreasoning"),
            filter.feed("<channel|>Answer<turn|>"),
            filter.finish()
        ].joined()

        #expect(output == "<think>\nreasoning</think>\nAnswer")
    }

    @Test("closes unfinished thought at finish")
    func closesUnfinishedThoughtAtFinish() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("<|channel>thought\nreasoning"),
            filter.finish()
        ].joined()

        #expect(output == "<think>\nreasoning</think>\n")
    }

    @Test("handles malformed bare channel opens")
    func handlesMalformedBareChannelOpens() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("<|channel>thoughtreasoning"),
            filter.feed("<channel|>Answer"),
            filter.finish()
        ].joined()

        #expect(output == "<think>\nreasoning</think>\nAnswer")
    }

    @Test("drops replacement characters at thought marker boundaries")
    func dropsReplacementCharactersAtThoughtMarkerBoundaries() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("<|channel>thought\nreasoning\u{FFFD}<chan"),
            filter.feed("nel|>"),
            filter.feed("\u{FFFD}Answer"),
            filter.finish()
        ].joined()

        #expect(output == "<think>\nreasoning</think>\nAnswer")
    }

    @Test("drops tool response protocol markers")
    func dropsToolResponseProtocolMarkers() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("before <|tool_response>{}"),
            filter.feed("<tool_response|> after"),
            filter.finish()
        ].joined()

        #expect(output == "before {} after")
    }

    @Test("preserves normal text")
    func preservesNormalText() {
        var filter = MLXGemma4StreamFilter()
        let output = [
            filter.feed("plain"),
            filter.feed(" answer"),
            filter.finish()
        ].joined()

        #expect(output == "plain answer")
    }
}
