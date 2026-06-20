#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
@testable import MLXFoundationModel
import Testing

@Suite("Foundation Models required tool argument grammar")
struct FMToolRequiredArgsTests {
    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    @Generable
    struct WeatherControlArguments {
        let city: String
        let count: Int
        let enabled: Bool
    }

    @Test("native EBNF grammars reject early close before required arguments")
    func nativeEBNFGrammarsRejectEarlyCloseBeforeRequiredArguments() throws {
        guard #available(macOS 27.0, iOS 27.0, visionOS 27.0, *) else {
            return
        }

        try Self.expectRejectsEarlyClose(style: .longCat, prefix: Self.longCatCityPrefix, allowed: ",")
        try Self.expectRejectsEarlyClose(style: .gemma, prefix: Self.gemmaCityPrefix, allowed: ",")
        try Self.expectRejectsEarlyClose(
            style: .minimaxM3,
            prefix: Self.miniMaxM3CityPrefix,
            allowed: #"]<]minimax[>[<count>"#
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static var weatherControlTool: Transcript.ToolDefinition {
        Transcript.ToolDefinition(
            name: "weather",
            description: "Read local weather for a city",
            parameters: WeatherControlArguments.generationSchema
        )
    }

    @available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
    private static func expectRejectsEarlyClose(
        style: MLXPromptStyle,
        prefix: String,
        allowed: String
    ) throws {
        let grammar = FMRequiredToolGrammarBuilder.grammar(
            from: [Self.weatherControlTool],
            promptStyle: style
        )

        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            grammar,
            prefix: prefix,
            allowed: allowed,
            rejected: "}"
        )
    }

    private static let longCatCityPrefix = #"<longcat_tool_call>{"arguments":{"city":"Berlin""#
    private static let gemmaCityPrefix = #"<|tool_call>call:weather{city:Berlin"#
    private static let miniMaxM3CityPrefix = miniMaxM3ToolPrefix +
        #"]<]minimax[>[<city>Berlin]<]minimax[>[</city>"#

    private static let miniMaxM3ToolPrefix = #"]<]minimax[>[<tool_call>"# +
        #"]<]minimax[>[<invoke name="weather">"#
}
#endif
