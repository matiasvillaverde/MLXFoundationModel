@testable import MLXFoundationModel
@testable import MLXLocalModels
import Testing

@Suite("Foundation Models tool JSON payload grammar")
struct FMToolJSONPayloadGrammarTests {
    @Test("required JSON payload keys can appear in reverse schema order")
    func requiredJSONPayloadKeysCanAppearInReverseSchemaOrder() throws {
        let grammar = Self.weatherPayloadGrammar()

        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            grammar,
            prefix: #"{"count":2"#,
            allowed: ",",
            rejected: "}"
        )
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            grammar,
            prefix: #"{"count":2,"city":"Berlin""#,
            allowed: "}",
            rejected: "]"
        )
    }

    @Test("required JSON payload keys reject early close in schema order")
    func requiredJSONPayloadKeysRejectEarlyCloseInSchemaOrder() throws {
        try NativeToolGrammarMaskTestSupport.expectAllowsOnlyAfterAcceptingPrefix(
            Self.weatherPayloadGrammar(),
            prefix: #"{"city":"Berlin""#,
            allowed: ",",
            rejected: "}"
        )
    }

    private static func weatherPayloadGrammar() -> GrammarSamplingConfiguration {
        let payload = FMToolJSONPayloadGrammar.payload(
            ruleName: "weather",
            schema: Self.weatherSchema()
        )
        let grammar = ([
            "root ::= \(payload.ruleName)"
        ] + payload.lines + FMToolJSONPayloadGrammar.supportRules)
            .joined(separator: "\n")
        return GrammarSamplingConfiguration(grammar: grammar)
    }

    private static func weatherSchema() -> [String: Any] {
        [
            "type": "object",
            "required": ["city", "count"],
            "properties": [
                "city": ["type": "string"],
                "count": ["type": "integer"],
                "enabled": ["type": "boolean"]
            ]
        ]
    }
}
