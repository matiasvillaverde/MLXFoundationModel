import Foundation

extension ToolSchema {
    static func normalizeContainer(
        _ value: Any,
        using schema: JSONObject
    ) -> Any {
        if let object = value as? JSONObject,
            isObjectLike(schema) {
            return normalizeNestedObject(object, using: schema)
        }
        if isArrayLike(schema),
            let values = arrayValue(from: value, schema: schema) {
            return values
        }
        return value
    }

    static func convertedValue(
        _ value: Any,
        as kind: String,
        schema: JSONObject
    ) -> Any? {
        switch kind {
        case "integer":
            integerValue(from: value)

        case "number":
            numberValue(from: value)

        case "boolean":
            booleanValue(from: value)

        case "object":
            objectValue(from: value, schema: schema)

        case "array":
            arrayValue(from: value, schema: schema)

        case "string":
            stringValue(from: value)

        default:
            nil
        }
    }

    static func integerValue(from value: Any) -> Int? {
        if let intValue = value as? Int {
            return intValue
        }
        if let string = value as? String {
            return Int(string.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    static func numberValue(from value: Any) -> Any? {
        if let double = value as? Double {
            return double
        }
        if let intValue = value as? Int {
            return intValue
        }
        if let string = value as? String {
            return Double(string.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return nil
    }

    static func booleanValue(from value: Any) -> Any? {
        if let bool = value as? Bool {
            return bool
        }
        guard let string = value as? String else {
            return nil
        }
        switch string.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "true", "1", "yes", "on":
            return true

        case "false", "0", "no", "off":
            return false

        default:
            return nil
        }
    }
}
