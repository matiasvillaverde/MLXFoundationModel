import Foundation

extension ToolSchema {
    static func schemaTypes(from schema: JSONObject) -> Set<String> {
        Set(schemaTypeOrder(from: schema))
    }

    static func schemaTypeOrder(from schema: JSONObject) -> [String] {
        var result: [String] = []
        appendTypes(from: schema["type"], to: &result)
        appendEnumTypes(from: schema["enum"], to: &result)
        appendConstType(from: schema["const"], to: &result)
        appendNullableType(from: schema["nullable"], to: &result)
        for key in ["anyOf", "oneOf", "allOf"] {
            guard let choices = schema[key] as? [Any] else {
                continue
            }
            appendTypes(from: choices, to: &result)
        }
        return result
    }

    static func directTypes(from value: Any?) -> Set<String> {
        if let string = value as? String {
            return [normalizedType(string)]
        }
        if let values = value as? [String] {
            return Set(values.map(normalizedType))
        }
        if let values = value as? [Any] {
            return Set(values.compactMap { $0 as? String }.map(normalizedType))
        }
        return []
    }

    static func inferredEnumTypes(from value: Any?) -> Set<String> {
        guard let values = value as? [Any] else {
            return []
        }
        return Set(values.map(enumType).map(normalizedType))
    }

    static func isNullLiteral(_ value: Any) -> Bool {
        if value is NSNull {
            return true
        }
        guard let string = value as? String else {
            return false
        }
        return ["null", "none", "nil"].contains(string.lowercased())
    }

    private static func appendTypes(from value: Any?, to result: inout [String]) {
        if let string = value as? String {
            appendUnique(normalizedType(string), to: &result)
            return
        }
        if let values = value as? [String] {
            for value in values {
                appendUnique(normalizedType(value), to: &result)
            }
            return
        }
        guard let values = value as? [Any] else {
            return
        }
        for value in values {
            if let string = value as? String {
                appendUnique(normalizedType(string), to: &result)
                continue
            }
            guard let child = value as? JSONObject else {
                continue
            }
            appendTypes(from: child, to: &result)
        }
    }

    private static func appendTypes(from choices: [Any], to result: inout [String]) {
        for choice in choices {
            guard let child = choice as? JSONObject else {
                continue
            }
            for type in schemaTypeOrder(from: child) {
                appendUnique(type, to: &result)
            }
        }
    }

    private static func appendTypes(from schema: JSONObject, to result: inout [String]) {
        appendTypes(from: schema["type"], to: &result)
        appendEnumTypes(from: schema["enum"], to: &result)
        appendConstType(from: schema["const"], to: &result)
        appendNullableType(from: schema["nullable"], to: &result)
        for key in ["anyOf", "oneOf", "allOf"] {
            guard let choices = schema[key] as? [Any] else {
                continue
            }
            appendTypes(from: choices, to: &result)
        }
    }

    private static func appendEnumTypes(from value: Any?, to result: inout [String]) {
        guard let values = value as? [Any] else {
            return
        }
        for value in values {
            appendUnique(normalizedType(enumType(value)), to: &result)
        }
    }

    private static func appendConstType(from value: Any?, to result: inout [String]) {
        guard let value else {
            return
        }
        appendUnique(normalizedType(enumType(value)), to: &result)
    }

    private static func appendNullableType(from value: Any?, to result: inout [String]) {
        guard let isNullable = value as? Bool, isNullable else {
            return
        }
        appendUnique("null", to: &result)
    }

    private static func appendUnique(_ type: String, to result: inout [String]) {
        guard !result.contains(type) else {
            return
        }
        result.append(type)
    }

    private static func enumType(_ value: Any) -> String {
        if value is NSNull {
            return "null"
        }
        if value is Bool {
            return "boolean"
        }
        if value is Int {
            return "integer"
        }
        if value is Double {
            return "number"
        }
        if value is String {
            return "string"
        }
        if value is [Any] {
            return "array"
        }
        if value is JSONObject {
            return "object"
        }
        return "string"
    }

    private static func normalizedType(_ type: String) -> String {
        let lowered = type.lowercased()
        if lowered == "int" || lowered.hasPrefix("uint") || lowered.hasPrefix("long") {
            return "integer"
        }
        if lowered == "float" || lowered.hasPrefix("num") {
            return "number"
        }
        if lowered == "bool" || lowered == "binary" {
            return "boolean"
        }
        if lowered == "arr" || lowered.hasPrefix("list") {
            return "array"
        }
        if lowered.hasPrefix("dict") {
            return "object"
        }
        if lowered == "str" || lowered == "text" || lowered == "enum" {
            return "string"
        }
        return lowered
    }
}
