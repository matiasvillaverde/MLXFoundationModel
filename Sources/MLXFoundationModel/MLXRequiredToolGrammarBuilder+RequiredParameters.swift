import Foundation

extension MLXRequiredToolGrammarBuilder {
    static func nativeParameterRules(
        from schema: [String: Any],
        parameterRuleName: String,
        format: FMNativeToolGrammarFormat
    ) -> [ParameterRule] {
        parameterSchemas(from: schema).enumerated().map { index, parameter in
            nativeParameterRule(
                index: index,
                parameter: parameter,
                parameterRuleName: parameterRuleName,
                format: format
            )
        }
    }

    static func gemmaParameterRules(
        from schema: [String: Any],
        parameterRuleName: String,
        format: FMNativeToolGrammarFormat
    ) -> [ParameterRule] {
        parameterSchemas(from: schema).enumerated().map { index, parameter in
            gemmaParameterRule(
                index: index,
                parameter: parameter,
                parameterRuleName: parameterRuleName,
                format: format
            )
        }
    }

    static func nativeFallbackLine(
        parameterRuleName: String,
        format: FMNativeToolGrammarFormat
    ) -> String {
        let fallbackParameter = format.parameter(
            parameterName: "native_parameter_name",
            valueRule: format.parameterFallbackRule
        )
        return "\(parameterRuleName) ::= \(fallbackParameter)"
    }

    static func parameterSequenceRuleName(for toolRuleName: String) -> String {
        "\(toolRuleName)_parameters"
    }

    static func parameterSequenceLines(
        ruleName: String,
        parameterRuleName: String,
        parameterRules: [ParameterRule],
        schema: [String: Any],
        separator: String
    ) -> [String] {
        let requiredRules = requiredParameterRules(parameterRules, schema: schema)
        guard !requiredRules.isEmpty else {
            return ["\(ruleName) ::= \(parameterRuleName)*"]
        }
        let requiredRuleName = "\(ruleName)_required"
        return ["\(ruleName) ::= \(requiredRuleName) \(parameterRuleName)*"] +
            requiredSequenceLines(
                ruleName: requiredRuleName,
                parameterRules: requiredRules,
                separator: separator
            )
    }

    static func gemmaArgumentsLines(
        ruleName: String,
        parameterRuleName: String,
        parameterRules: [ParameterRule],
        schema: [String: Any]
    ) -> [String] {
        let requiredRules = requiredParameterRules(parameterRules, schema: schema)
        guard !requiredRules.isEmpty else {
            return ["\(ruleName) ::= \"{\" (\(parameterRuleName) (\",\" \(parameterRuleName))*)? \"}\""]
        }
        let requiredRuleName = "\(ruleName)_required"
        return ["\(ruleName) ::= \"{\" \(requiredRuleName) (\",\" \(parameterRuleName))* \"}\""] +
            requiredSequenceLines(
                ruleName: requiredRuleName,
                parameterRules: requiredRules,
                separator: #" "," "#
            )
    }

    private static func requiredSequenceLines(
        ruleName: String,
        parameterRules: [ParameterRule],
        separator: String
    ) -> [String] {
        let alternatives = ToolSchema.requiredOrderPermutations(parameterRules)
            .map { permutation in
                permutation.map(\.name).joined(separator: separator)
            }
            .joined(separator: " | ")
        return ["\(ruleName) ::= \(alternatives)"]
    }

    private static func requiredParameterRules(
        _ rules: [ParameterRule],
        schema: [String: Any]
    ) -> [ParameterRule] {
        let requiredKeys = Set(ToolSchema.requiredPropertyNames(from: schema))
        guard !requiredKeys.isEmpty else {
            return []
        }
        return rules.filter { requiredKeys.contains($0.key) }
    }
}
