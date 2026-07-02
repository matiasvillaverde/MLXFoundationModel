import Foundation
import MLXLocalModels

enum MLXRequiredToolGrammarBuilder {
    private struct ToolRule {
        let name: String
        let lines: [String]
    }

    struct ParameterSchema {
        let key: String
        let schema: [String: Any]
    }

    struct ParameterRule {
        let key: String
        let name: String
        let line: String
    }

    static func grammar(
        from definitions: [MLXBridgeToolDefinition],
        promptStyle: MLXPromptStyle
    ) -> GrammarSamplingConfiguration {
        guard let format = FMNativeToolGrammarFormat(promptStyle: promptStyle) else {
            return .jsonSchema(requiredToolCallSchema(from: definitions))
        }
        if let grammar = structuralTagGrammar(from: definitions, format: format) {
            return grammar
        }
        return GrammarSamplingConfiguration(grammar: nativeGrammar(from: definitions, format: format))
    }

    static func requiredToolCallSchema(from definitions: [MLXBridgeToolDefinition]) -> String {
        let choices = definitions.map(toolCallChoiceSchema)
        if choices.count == 1, let schema = choices.first {
            return jsonString(from: schema)
        }
        return jsonString(from: ["oneOf": choices])
    }

    private static func nativeGrammar(
        from definitions: [MLXBridgeToolDefinition],
        format: FMNativeToolGrammarFormat
    ) -> String {
        let rules = definitions.enumerated().map { index, definition in
            nativeToolRule(index: index, definition: definition, format: format)
        }
        let ruleNames = rules.map(\.name).joined(separator: " | ")
        return ([
            "root ::= native_tool_call",
            "native_tool_call ::= \(ruleNames)"
        ] + rules.flatMap(\.lines) + FMToolJSONPayloadGrammar.supportRules + [
            #"native_parameter_name ::= [A-Za-z_] [A-Za-z0-9_-]*"#,
            #"native_xml_text ::= [^<]*"#,
            #"native_bracket_text ::= [^\]]*"#,
            #"native_scalar_text ::= [A-Za-z0-9_ .:/+\-]*"#
        ])
        .joined(separator: "\n")
    }

    private static func nativeToolRule(
        index: Int,
        definition: MLXBridgeToolDefinition,
        format: FMNativeToolGrammarFormat
    ) -> ToolRule {
        switch format.ruleKind {
        case .functionGemma:
            return gemmaRule(index: index, definition: definition, isFunctionGemma: true)

        case .gemma:
            return gemmaRule(index: index, definition: definition, isFunctionGemma: false)

        case .jsonEnvelope:
            return simpleJSONRule(index: index, definition: definition, format: format)

        case .xmlParameters:
            return xmlParameterRule(index: index, definition: definition, format: format)
        }
    }

    private static func simpleJSONRule(
        index: Int,
        definition: MLXBridgeToolDefinition,
        format: FMNativeToolGrammarFormat
    ) -> ToolRule {
        let ruleName = nativeRuleName(index: index, definition: definition, format: format)
        let payload = FMToolJSONPayloadGrammar.payload(
            ruleName: ruleName,
            schema: schemaValue(for: definition)
        )
        return ToolRule(
            name: ruleName,
            lines: [
                """
                \(ruleName) ::= \(format.simpleJSONEnvelope(
                    toolName: definition.name,
                    payloadRule: payload.ruleName
                ))
                """
            ] + payload.lines
        )
    }

    private static func xmlParameterRule(
        index: Int,
        definition: MLXBridgeToolDefinition,
        format: FMNativeToolGrammarFormat
    ) -> ToolRule {
        let ruleName = nativeRuleName(index: index, definition: definition, format: format)
        let parameterRuleName = "\(ruleName)_parameter"
        let schema = expandedSchema(for: definition)
        let parameterRules = nativeParameterRules(
            from: schema,
            parameterRuleName: parameterRuleName,
            format: format
        )
        let fallbackLine = nativeFallbackLine(parameterRuleName: parameterRuleName, format: format)
        return ToolRule(
            name: ruleName,
            lines: [
                """
                \(ruleName) ::= \(format.prefix(toolName: definition.name)) \
                \(parameterSequenceRuleName(for: ruleName)) \(format.suffix)
                """
            ] + parameterSequenceLines(
                ruleName: parameterSequenceRuleName(for: ruleName),
                parameterRuleName: parameterRuleName,
                parameterRules: parameterRules,
                schema: schema,
                separator: " "
            ) + parameterDispatchLines(
                parameterRuleName: parameterRuleName,
                parameterRules: parameterRules,
                fallbackLine: fallbackLine
            )
        )
    }

    private static func gemmaRule(
        index: Int,
        definition: MLXBridgeToolDefinition,
        isFunctionGemma: Bool
    ) -> ToolRule {
        let format: FMNativeToolGrammarFormat = isFunctionGemma ? .functionGemma : .gemma
        let ruleName = nativeRuleName(index: index, definition: definition, format: format)
        let parameterRuleName = "\(ruleName)_parameter"
        let schema = expandedSchema(for: definition)
        let parameterRules = gemmaParameterRules(
            from: schema,
            parameterRuleName: parameterRuleName,
            format: format
        )
        let fallbackLine = "\(parameterRuleName) ::= native_parameter_name \":\" native_scalar_text"
        let argumentsRuleName = "\(ruleName)_arguments"
        return ToolRule(
            name: ruleName,
            lines: [
                "\(ruleName) ::= \(format.prefix(toolName: definition.name)) "
                    + "\(argumentsRuleName) \(format.suffix)"
            ] + gemmaArgumentsLines(
                ruleName: argumentsRuleName,
                parameterRuleName: parameterRuleName,
                parameterRules: parameterRules,
                schema: schema
            ) + parameterDispatchLines(
                parameterRuleName: parameterRuleName,
                parameterRules: parameterRules,
                fallbackLine: fallbackLine
            )
        )
    }

    private static func parameterDispatchLines(
        parameterRuleName: String,
        parameterRules: [ParameterRule],
        fallbackLine: String
    ) -> [String] {
        guard !parameterRules.isEmpty else {
            return [fallbackLine]
        }
        let alternatives = parameterRules.map(\.name).joined(separator: " | ")
        return ["\(parameterRuleName) ::= \(alternatives)"] + parameterRules.map(\.line)
    }
}
