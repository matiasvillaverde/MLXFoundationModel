#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FMRequiredToolGrammarBuilder {
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
        from definitions: [Transcript.ToolDefinition],
        promptStyle: MLXPromptStyle
    ) -> GrammarSamplingConfiguration {
        guard let format = FMNativeToolGrammarFormat(promptStyle: promptStyle) else {
            return .jsonSchema(
                FoundationModelsToolSchemaBuilder.requiredToolCallSchema(from: definitions)
            )
        }
        if let grammar = structuralTagGrammar(from: definitions, format: format) {
            return grammar
        }
        return GrammarSamplingConfiguration(grammar: nativeGrammar(from: definitions, format: format))
    }

    private static func nativeGrammar(
        from definitions: [Transcript.ToolDefinition],
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
        definition: Transcript.ToolDefinition,
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
        definition: Transcript.ToolDefinition,
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
        definition: Transcript.ToolDefinition,
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
        definition: Transcript.ToolDefinition,
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

    static func nativeParameterRule(
        index: Int,
        parameter: ParameterSchema,
        parameterRuleName: String,
        format: FMNativeToolGrammarFormat
    ) -> ParameterRule {
        let ruleName = "\(parameterRuleName)_property_\(index)_\(identifierFragment(parameter.key))"
        let valueRule = FMToolParameterValueGrammar.valueRule(
            for: parameter.schema,
            fallback: format.parameterFallbackRule
        )
        let parameterRule = format.parameter(
            parameterName: literal(parameter.key),
            valueRule: valueRule,
            stringEncoded: stringEncodedParameterValue(parameter.schema, format: format)
        )
        return ParameterRule(
            key: parameter.key,
            name: ruleName,
            line: "\(ruleName) ::= \(parameterRule)"
        )
    }

    static func gemmaParameterRule(
        index: Int,
        parameter: ParameterSchema,
        parameterRuleName: String,
        format: FMNativeToolGrammarFormat
    ) -> ParameterRule {
        let ruleName = "\(parameterRuleName)_property_\(index)_\(identifierFragment(parameter.key))"
        let valueRule = FMToolParameterValueGrammar.valueRule(
            for: parameter.schema,
            fallback: format.parameterFallbackRule
        )
        return ParameterRule(
            key: parameter.key,
            name: ruleName,
            line: "\(ruleName) ::= \(literal(parameter.key)) \":\" \(valueRule)"
        )
    }

    private static func nativeRuleName(
        index: Int,
        definition: Transcript.ToolDefinition,
        format: FMNativeToolGrammarFormat
    ) -> String {
        let formatName = identifierFragment(format.rawValue).lowercased()
        return "tool_\(index)_\(formatName)_\(identifierFragment(definition.name))"
    }

    private static func literal(_ value: String) -> String {
        #""\#(escapedEBNFLiteral(value))""#
    }

    private static func expandedSchema(for definition: Transcript.ToolDefinition) -> [String: Any] {
        let root = schemaValue(for: definition)
        let expanded = ToolSchema.expandedSchema(root, root: root)
        return ToolSchema.schemaWithMergedBranchProperties(expanded)
    }

    static func parameterSchemas(from schema: [String: Any]) -> [ParameterSchema] {
        guard let properties = schema["properties"] as? [String: Any] else {
            return []
        }
        return properties.keys.sorted().compactMap { key in
            guard let schema = properties[key] as? [String: Any] else {
                return nil
            }
            return ParameterSchema(key: key, schema: schema)
        }
    }

    private static func schemaValue(for definition: Transcript.ToolDefinition) -> [String: Any] {
        let schema = FoundationModelsSchemaSupport.jsonSchemaString(from: definition.parameters)
        guard
            let object = FoundationModelsSchemaSupport.jsonValue(from: schema) as? [String: Any],
            !object.isEmpty
        else {
            return [:]
        }
        return object
    }

    private static func identifierFragment(_ value: String) -> String {
        let mapped = value.map { character in
            character.isLetter || character.isNumber ? character : "_"
        }
        let text = String(mapped)
        return text.isEmpty ? "tool" : text
    }
}
#endif
