@testable import MLXFoundationModel
import Testing

@Suite("MLX tool call stream filter")
struct MLXToolCallStreamFilterTests {
    @Test("streams prose while suppressing split XML tool envelopes")
    func streamsProseWhileSuppressingSplitXMLToolEnvelopes() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("The answer is before <too"),
            filter.feed(#"l_call>{"name":"weather","arguments":{"city":"Berlin"}}</tool_call> after"#),
            filter.finish()
        ].joined()

        #expect(output == "The answer is before  after")
    }

    @Test("suppresses Mistral marker while preserving trailing prose")
    func suppressesMistralMarkerWhilePreservingTrailingProse() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before [TOOL"),
            filter.feed(#"_CALLS]weather[ARGS]{"city":"Berlin"} trailing"#),
            filter.finish()
        ].joined()

        #expect(output == "before  trailing")
    }

    @Test("waits for split Mistral JSON before streaming trailing prose")
    func waitsForSplitMistralJSONBeforeStreamingTrailingProse() {
        var filter = MLXToolCallStreamFilter()
        let chunks = [
            filter.feed(#"before [TOOL_CALLS]weather[ARGS]{"city":"Ber"#),
            filter.feed(#"lin","note":"brace } inside string"} after"#),
            filter.finish()
        ]

        #expect(chunks[0] == "before ")
        #expect(chunks[1] == " after")
        #expect(chunks[2].isEmpty)
    }

    @Test("suppresses Mistral JSON array tool calls")
    func suppressesMistralJSONArrayToolCalls() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed(#"before [TOOL_CALLS][{"name":"weather","arguments":{"city":"Berlin"}}]"#),
            filter.feed(" after"),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("suppresses parseable bracket tool call while preserving trailing prose")
    func suppressesParseableBracketToolCallWhilePreservingTrailingProse() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed(#"before [Calling tool: weather({"city":"Berlin"})] after"#),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("waits for split bracket tool call before streaming trailing prose")
    func waitsForSplitBracketToolCallBeforeStreamingTrailingProse() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed(#"before [Calling tool: weather({"city":"Ber"#),
            filter.feed(#"lin","note":"paren ) inside string"})] after"#),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("preserves literal bracket tool-looking prose")
    func preservesLiteralBracketToolLookingProse() {
        var filter = MLXToolCallStreamFilter()
        let text = "literal [Calling tool: maybe later] and keep going"
        let output = [
            filter.feed(text),
            filter.finish()
        ].joined()

        #expect(output == text)
    }

    @Test("drops unfinished bracket tool call at finish")
    func dropsUnfinishedBracketToolCallAtFinish() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed(#"before [Tool call: weather({"city":"Berlin"}"#),
            filter.finish()
        ].joined()

        #expect(output == "before ")
    }

    @Test("suppresses MiniMax M3 namespace-token tool envelopes")
    func suppressesMiniMaxM3NamespaceTokenToolEnvelopes() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before ]<]minimax[>[<tool"),
            filter.feed(#"_call>]<]minimax[>[<invoke name="weather">"#),
            filter.feed("]<]minimax[>[<city>Berlin]<]minimax[>[</city>"),
            filter.feed("]<]minimax[>[</invoke>]<]minimax[>[</tool_call> after"),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("suppresses bare Kimi tool call envelope")
    func suppressesBareKimiToolCallEnvelope() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before <|tool_call_beg"),
            filter.feed(#"in|>functions.weather:0<|tool_call_argument_begin|>{"city":"Berlin"}"#),
            filter.feed("<|tool_call_end|> after"),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("suppresses namespaced XML tool envelope split before colon")
    func suppressesNamespacedXMLToolEnvelopeSplitBeforeColon() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before <foo-"),
            filter.feed(#"bar:too"#),
            filter.feed(#"l_call><invoke name="weather"></invoke></foo-bar:tool_call> after"#),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("suppresses bare GLM tool call while waiting for additional arguments")
    func suppressesBareGLMToolCallWhileWaitingForAdditionalArguments() {
        var filter = MLXToolCallStreamFilter(toolNames: ["weather"])
        let chunks = [
            filter.feed("before wea"),
            filter.feed(#"ther<arg_key>city</arg_key><arg_value>"Berlin"</arg_value>"#),
            filter.feed(#"<arg_key>count</arg_key><arg_value>2</arg_value> after"#),
            filter.finish()
        ]
        let suppressed = filter.takeCompletedSuppressedTexts()

        #expect(chunks == ["before ", "", " after", ""])
        #expect(suppressed.count == 1)
        #expect(suppressed[0].contains(#"weather<arg_key>city</arg_key>"#))
        #expect(suppressed[0].contains(#"<arg_key>count</arg_key>"#))
    }

    @Test("preserves unknown bare GLM-shaped text")
    func preservesUnknownBareGLMShapedText() {
        var filter = MLXToolCallStreamFilter(toolNames: ["weather"])
        let text = #"other<arg_key>city</arg_key><arg_value>"Berlin"</arg_value>"#
        let output = [
            filter.feed(text),
            filter.finish()
        ].joined()

        #expect(output == text)
    }

    @Test("preserves split non-tool namespaced literal")
    func preservesSplitNonToolNamespacedLiteral() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("literal <alpha"),
            filter.feed(":beta"),
            filter.finish()
        ].joined()

        #expect(output == "literal <alpha:beta")
    }

    @Test("drops split orphan Gemma close marker")
    func dropsSplitOrphanGemmaCloseMarker() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before <tool"),
            filter.feed("_call|> after"),
            filter.finish()
        ].joined()

        #expect(output == "before  after")
    }

    @Test("drops unfinished tool marker at end")
    func dropsUnfinishedToolMarkerAtEnd() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("before <|tool"),
            filter.finish()
        ].joined()

        #expect(output == "before ")
    }

    @Test("flushes literal unfinished XML text at end")
    func flushesLiteralUnfinishedXMLTextAtEnd() {
        var filter = MLXToolCallStreamFilter()
        let output = [
            filter.feed("literal <alpha"),
            filter.finish()
        ].joined()

        #expect(output == "literal <alpha")
    }

    @Test("preserves normal prose across chunks")
    func preservesNormalProseAcrossChunks() {
        var filter = MLXToolCallStreamFilter()
        let input = "This is a normal answer with enough text to flush while streaming."
        let output = [
            filter.feed(String(input.prefix(30))),
            filter.feed(String(input.dropFirst(30))),
            filter.finish()
        ].joined()

        #expect(output == input)
    }
}
