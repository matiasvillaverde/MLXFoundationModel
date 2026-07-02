import Foundation

extension MLXRequiredToolGrammarBuilder {
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

    static func parameterSchemas(from schema: [String: Any]) -> [ParameterSchema] {
        guard let properties = schema["properties"] as? [String: Any] else {
            return []
        }
        return properties.keys.sorted().compactMap { key in
            guard let propertySchema = properties[key] as? [String: Any] else {
                return nil
            }
            return ParameterSchema(key: key, schema: propertySchema)
        }
    }

    static func expandedSchema(for definition: MLXBridgeToolDefinition) -> [String: Any] {
        let root = schemaValue(for: definition)
        let expanded = ToolSchema.expandedSchema(root, root: root)
        return ToolSchema.schemaWithMergedBranchProperties(expanded)
    }

    static func schemaValue(for definition: MLXBridgeToolDefinition) -> [String: Any] {
        jsonValue(from: definition.parametersJSONSchema) as? [String: Any] ?? [:]
    }

    static func nativeRuleName(
        index: Int,
        definition: MLXBridgeToolDefinition,
        format: FMNativeToolGrammarFormat
    ) -> String {
        let formatName = identifierFragment(format.rawValue).lowercased()
        return "tool_\(index)_\(formatName)_\(identifierFragment(definition.name))"
    }

    static func toolCallChoiceSchema(
        for definition: MLXBridgeToolDefinition
    ) -> [String: Any] {
        [
            "additionalProperties": false,
            "properties": [
                "arguments": schemaValue(for: definition),
                "tool_name": ["enum": [definition.name]]
            ],
            "required": [
                "tool_name",
                "arguments"
            ],
            "type": "object"
        ]
    }

    static func jsonValue(from string: String) -> Any {
        guard
            let data = string.data(using: .utf8),
            let value = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        else {
            return string
        }
        return value
    }

    static func jsonString(from value: Any) -> String {
        guard
            JSONSerialization.isValidJSONObject(value),
            let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }

    static func literal(_ value: String) -> String {
        #""\#(escapedEBNFLiteral(value))""#
    }

    static func identifierFragment(_ value: String) -> String {
        let mapped = value.map { character in
            character.isLetter || character.isNumber ? character : "_"
        }
        let text = String(mapped)
        return text.isEmpty ? "tool" : text
    }
}
