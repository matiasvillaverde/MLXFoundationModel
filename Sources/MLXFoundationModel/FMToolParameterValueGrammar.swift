import Foundation

enum FMToolParameterValueGrammar {
    typealias JSONObject = [String: Any]

    static func valueRule(for schema: JSONObject, fallback: String) -> String {
        let literalRules = literalChoices(for: schema)
            .map(literal)
            .joined(separator: " | ")
        if !literalRules.isEmpty {
            return literalRules
        }

        let types = ToolSchema.schemaTypeOrder(from: schema)
        let rules = types.isEmpty ? [fallback] : types.map { ruleName(for: $0, fallback: fallback) }
        return removingDuplicates(from: rules).joined(separator: " | ")
    }

    private static func ruleName(for type: String, fallback: String) -> String {
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
            return fallback

        default:
            return fallback
        }
    }

    private static func literalChoices(for schema: JSONObject) -> [String] {
        var values: [Any] = []
        if schema.keys.contains("const"),
            let value = schema["const"] {
            values.append(value)
        }
        if let enumValues = schema["enum"] as? [Any] {
            values.append(contentsOf: enumValues)
        }
        return removingDuplicates(from: values.compactMap(textLiteral))
    }

    private static func textLiteral(_ value: Any) -> String? {
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
        return value as? String
    }

    private static func literal(_ value: String) -> String {
        #""\#(escapedEBNFLiteral(value))""#
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
