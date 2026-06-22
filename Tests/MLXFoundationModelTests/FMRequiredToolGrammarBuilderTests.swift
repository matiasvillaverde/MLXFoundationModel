#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite("Foundation Models required tool grammar")
struct FMRequiredToolGrammarBuilderTests {
    private struct NativeGrammarFixture {
        let style: MLXPromptStyle
        let nativeStart: String
        let kind: GrammarConstraintKind
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct WeatherArguments {
        let city: String
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct WeatherCountArguments {
        let count: Int
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct WeatherEnabledArguments {
        let enabled: Bool
    }

    @Test("native prompt styles constrain required tool calls to native envelopes")
    func nativePromptStylesConstrainRequiredToolCallsToNativeEnvelopes() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        for fixture in Self.nativeFixtures {
            let grammar = FMRequiredToolGrammarBuilder.grammar(
                from: [Self.weatherTool],
                promptStyle: fixture.style
            )

            #expect(grammar.kind == fixture.kind)
            #expect(!grammar.grammar.contains(#""tool_name""#))
            try NativeToolGrammarMaskTestSupport.expectRejectsJSONStart(
                grammar,
                nativeStart: fixture.nativeStart
            )
        }
    }

    @Test("generic styles keep the schema required-tool grammar")
    func genericStylesKeepSchemaRequiredToolGrammar() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let grammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherTool],
            promptStyle: .chatML
        )

        #expect(grammar.kind == .jsonSchema)
        #expect(grammar.grammar.contains(#""tool_name""#))
        #expect(grammar.grammar.contains("weather"))
    }

    @Test("JSON envelope native grammars constrain payload keys from the tool schema")
    func jsonEnvelopeNativeGrammarsConstrainPayloadKeysFromToolSchema() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let grammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherTool],
            promptStyle: .longCat
        )

        #expect(grammar.grammar.contains("weather"))
        #expect(grammar.grammar.contains(#""\"city\"""#))
        #expect(!grammar.grammar.contains("native_json_payload ::= [^<]*"))
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            grammar,
            prefix: #"<longcat_tool_call>{"arguments":"#,
            allowed: #"{"city":"#,
            rejected: #"{"unknown":"#
        )
    }

    @Test("XML native grammars constrain scalar values from the tool schema")
    func xmlNativeGrammarsConstrainScalarValuesFromToolSchema() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let countGrammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherCountTool],
            promptStyle: .qwenXML
        )

        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            countGrammar,
            prefix: #"<tool_call><function=weather><parameter=count>"#,
            allowed: "2",
            rejected: "Berlin"
        )

        let enabledGrammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherEnabledTool],
            promptStyle: .qwenXML
        )
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            enabledGrammar,
            prefix: #"<tool_call><function=weather><parameter=enabled>"#,
            allowed: "true",
            rejected: "2"
        )
    }

    @Test("DeepSeek DSML native grammars constrain typed parameter values")
    func deepSeekDSMLNativeGrammarsConstrainTypedParameterValues() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let countGrammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherCountTool],
            promptStyle: .deepSeekDSML
        )

        #expect(countGrammar.kind == .structuralTag)
        #expect(countGrammar.grammar.contains(#""style":"deepseek_xml""#))
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            countGrammar,
            prefix: Self.deepSeekCountPrefix,
            allowed: "2",
            rejected: "Berlin"
        )

        let enabledGrammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherEnabledTool],
            promptStyle: .deepSeekDSML
        )
        #expect(enabledGrammar.kind == .structuralTag)
        #expect(enabledGrammar.grammar.contains(#""style":"deepseek_xml""#))
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            enabledGrammar,
            prefix: Self.deepSeekEnabledPrefix,
            allowed: "true",
            rejected: "2"
        )
    }

    @Test("Gemma native grammars constrain scalar values from the tool schema")
    func gemmaNativeGrammarsConstrainScalarValuesFromToolSchema() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        let grammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherCountTool],
            promptStyle: .gemma
        )

        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            grammar,
            prefix: #"<|tool_call>call:weather{count:"#,
            allowed: "2",
            rejected: "Berlin"
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherArguments.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherCountTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherCountArguments.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherEnabledTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherEnabledArguments.generationSchema
        )
    }

    private static let nativeFixtures: [NativeGrammarFixture] = [
        .init(
            style: .cohereAction,
            nativeStart: #"<|START_ACTION|>{"tool_name":"weather","parameters":"#,
            kind: .ebnf
        ),
        .init(
            style: .deepSeekDSML,
            nativeStart: Self.deepSeekNativeStart,
            kind: .structuralTag
        ),
        .init(
            style: .functionGemma,
            nativeStart: #"<start_function_call>call:weather"#,
            kind: .ebnf
        ),
        .init(
            style: .gemma,
            nativeStart: #"<|tool_call>call:weather"#,
            kind: .ebnf
        ),
        .init(
            style: .glmXML,
            nativeStart: #"<tool_call>weather"#,
            kind: .structuralTag
        ),
        .init(
            style: .harmony,
            nativeStart: #"<|start|>assistant to=functions.weather<|channel|>commentary<|message|>"#,
            kind: .ebnf
        ),
        .init(
            style: .kimiK2,
            nativeStart: Self.kimiNativeStart,
            kind: .ebnf
        ),
        .init(
            style: .longCat,
            nativeStart: #"<longcat_tool_call>{"arguments":"#,
            kind: .ebnf
        ),
        .init(
            style: .minimaxM3,
            nativeStart: #"]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="weather">"#,
            kind: .ebnf
        ),
        .init(
            style: .minimaxXML,
            nativeStart: #"<minimax:tool_call><invoke name="weather">"#,
            kind: .structuralTag
        ),
        .init(
            style: .mistralToolCall,
            nativeStart: #"[TOOL_CALLS]weather[ARGS]"#,
            kind: .ebnf
        ),
        .init(
            style: .qwenXML,
            nativeStart: #"<tool_call><function=weather>"#,
            kind: .structuralTag
        )
    ]

    private static let kimiNativeStart = "<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:0"
        + "<|tool_call_argument_begin|>"

    private static let deepSeekNativeStart = "\n\n<｜DSML｜tool_calls>\n"
        + #"<｜DSML｜invoke name="weather">"#
        + "\n"

    private static let deepSeekCountPrefix = deepSeekNativeStart
        + #"<｜DSML｜parameter name="count" string="false">"#

    private static let deepSeekEnabledPrefix = deepSeekNativeStart
        + #"<｜DSML｜parameter name="enabled" string="false">"#
}
#endif
