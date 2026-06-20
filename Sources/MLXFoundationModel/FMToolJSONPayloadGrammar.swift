import Foundation

enum FMToolJSONPayloadGrammar {
    typealias JSONObject = [String: Any]

    struct Payload {
        let ruleName: String
        let lines: [String]
    }

    static let supportRules = [
        #"fm_json_ws ::= [ \t\n\r]*"#,
        #"fm_json_string ::= "\"" fm_json_string_char* "\"""#,
        #"fm_json_string_char ::= [^"\\] | "\\" ["\\/bfnrt]"#,
        #"fm_json_integer ::= "-" [0-9] [0-9]* | [0-9] [0-9]*"#,
        #"fm_json_number ::= fm_json_integer "." [0-9] [0-9]* | fm_json_integer"#,
        #"fm_json_boolean ::= "true" | "false""#,
        #"fm_json_object_lax ::= "{" [^}]* "}""#,
        #"fm_json_array_lax ::= "[" [^\]]* "]""#,
        #"fm_json_value ::= fm_json_string | fm_json_number | fm_json_boolean | "null" | "# +
            #"fm_json_object_lax | fm_json_array_lax"#
    ]

    static func payload(ruleName: String, schema: Any) -> Payload {
        let payloadRuleName = "\(ruleName)_json_payload"
        let root = expandedPayloadSchema(from: schema)
        let properties = propertySchemas(from: root)
        guard !properties.isEmpty else {
            return Payload(ruleName: payloadRuleName, lines: [
                "\(payloadRuleName) ::= fm_json_object_lax"
            ])
        }

        let pairRuleName = "\(payloadRuleName)_pair"
        let propertyRules = properties.enumerated().map { index, property in
            propertyRule(
                index: index,
                key: property.key,
                schema: property.schema,
                payloadRuleName: payloadRuleName
            )
        }
        let propertyRuleNames = propertyRules.map(\.name).joined(separator: " | ")
        let rootLines = payloadRootLines(
            payloadRuleName: payloadRuleName,
            pairRuleName: pairRuleName,
            propertyRules: propertyRules,
            schema: root
        )
        let lines = rootLines + [
            "\(pairRuleName) ::= \(propertyRuleNames)"
        ] + propertyRules.map(\.line)

        return Payload(ruleName: payloadRuleName, lines: lines)
    }

    private struct PropertySchema {
        let key: String
        let schema: JSONObject
    }

    private struct PropertyRule {
        let key: String
        let name: String
        let line: String
    }

    private static func propertySchemas(from schema: JSONObject) -> [PropertySchema] {
        guard let properties = schema["properties"] as? JSONObject else {
            return []
        }
        return properties.keys.sorted().compactMap { key in
            guard let schema = properties[key] as? JSONObject else {
                return nil
            }
            return PropertySchema(key: key, schema: schema)
        }
    }

    private static func expandedPayloadSchema(from schema: Any) -> JSONObject {
        guard let root = schema as? JSONObject else {
            return [:]
        }
        let expanded = ToolSchema.expandedSchema(root, root: root)
        return ToolSchema.schemaWithMergedBranchProperties(expanded)
    }

    private static func propertyRule(
        index: Int,
        key: String,
        schema: JSONObject,
        payloadRuleName: String
    ) -> PropertyRule {
        let name = "\(payloadRuleName)_property_\(index)_\(identifierFragment(key))"
        let valueRule = jsonValueRule(for: schema)
        return PropertyRule(
            key: key,
            name: name,
            line: "\(name) ::= \(literal(jsonStringLiteral(key))) fm_json_ws \":\" fm_json_ws \(valueRule)"
        )
    }

    private static func payloadRootLines(
        payloadRuleName: String,
        pairRuleName: String,
        propertyRules: [PropertyRule],
        schema: JSONObject
    ) -> [String] {
        let requiredRules = requiredPropertyRules(propertyRules, schema: schema)
        guard !requiredRules.isEmpty else {
            return [
                """
            \(payloadRuleName) ::= "{" fm_json_ws \
            (\(pairRuleName) (fm_json_ws "," fm_json_ws \(pairRuleName))*)? \
            fm_json_ws "}"
            """
            ]
        }
        let requiredRuleName = "\(payloadRuleName)_required"
        return [
            """
        \(payloadRuleName) ::= "{" fm_json_ws \
        \(requiredRuleName) \
        (fm_json_ws "," fm_json_ws \(pairRuleName))* fm_json_ws "}"
        """
        ] + requiredSequenceLines(
            ruleName: requiredRuleName,
            rules: requiredRules
        )
    }

    private static func requiredSequenceLines(
        ruleName: String,
        rules: [PropertyRule]
    ) -> [String] {
        let alternatives = ToolSchema.requiredOrderPermutations(rules)
            .map { permutation in
                permutation.map(\.name).joined(separator: #" fm_json_ws "," fm_json_ws "#)
            }
            .joined(separator: " | ")
        return ["\(ruleName) ::= \(alternatives)"]
    }

    private static func requiredPropertyRules(
        _ rules: [PropertyRule],
        schema: JSONObject
    ) -> [PropertyRule] {
        let requiredKeys = Set(ToolSchema.requiredPropertyNames(from: schema))
        guard !requiredKeys.isEmpty else {
            return []
        }
        return rules.filter { requiredKeys.contains($0.key) }
    }

    private static func jsonValueRule(for schema: JSONObject) -> String {
        let literalRules = jsonLiteralChoices(for: schema).map(literal).joined(separator: " | ")
        if !literalRules.isEmpty {
            return literalRules
        }

        let types = ToolSchema.schemaTypeOrder(from: schema)
        let rules = types.isEmpty ? ["fm_json_value"] : types.map(jsonRuleName(for:))
        return removingDuplicates(from: rules).joined(separator: " | ")
    }

    private static func jsonRuleName(for type: String) -> String {
        switch type {
        case "boolean":
            return "fm_json_boolean"

        case "integer":
            return "fm_json_integer"

        case "number":
            return "fm_json_number"

        case "null":
            return #""null""#

        case "object":
            return "fm_json_object_lax"

        case "array":
            return "fm_json_array_lax"

        case "string":
            return "fm_json_string"

        default:
            return "fm_json_value"
        }
    }

    private static func jsonLiteralChoices(for schema: JSONObject) -> [String] {
        var values: [Any] = []
        if schema.keys.contains("const"),
            let value = schema["const"] {
            values.append(value)
        }
        if let enumValues = schema["enum"] as? [Any] {
            values.append(contentsOf: enumValues)
        }
        return removingDuplicates(from: values.compactMap(jsonLiteral))
    }

    private static func jsonLiteral(_ value: Any) -> String? {
        if value is NSNull {
            return "null"
        }
        if let bool = value as? Bool {
            return bool ? "true" : "false"
        }
        if let int = value as? Int {
            return "\(int)"
        }
        if let int64 = value as? Int64 {
            return "\(int64)"
        }
        if let uint = value as? UInt {
            return "\(uint)"
        }
        if let double = value as? Double,
            double.isFinite {
            return "\(double)"
        }
        if let float = value as? Float,
            float.isFinite {
            return "\(float)"
        }
        if let string = value as? String {
            return jsonStringLiteral(string)
        }
        return nil
    }

    private static func jsonStringLiteral(_ value: String) -> String {
        guard
            let data = try? JSONSerialization.data(withJSONObject: [value], options: []),
            let string = String(data: data, encoding: .utf8),
            string.count >= 2
        else {
            return #""\#(value)""#
        }
        return String(string.dropFirst().dropLast())
    }

    private static func literal(_ value: String) -> String {
        #""\#(escapedEBNFLiteral(value))""#
    }

    private static func identifierFragment(_ value: String) -> String {
        let mapped = value.map { character in
            character.isLetter || character.isNumber ? character : "_"
        }
        let text = String(mapped)
        return text.isEmpty ? "property" : text
    }

    private static func escapedEBNFLiteral(_ value: String) -> String {
        value.reduce(into: "") { result, character in
            switch character {
            case "\"":
                result += #"\""#

            case "\\":
                result += #"\\"#

            case "\n":
                result += #"\n"#

            case "\r":
                result += #"\r"#

            case "\t":
                result += #"\t"#

            default:
                result.append(character)
            }
        }
    }

    private static func removingDuplicates<Value: Hashable>(from values: [Value]) -> [Value] {
        var seen: Set<Value> = []
        return values.filter { seen.insert($0).inserted }
    }
}
