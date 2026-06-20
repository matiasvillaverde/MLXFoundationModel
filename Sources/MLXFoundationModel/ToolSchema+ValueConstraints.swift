import Foundation

extension ToolSchema {
    static func stringConstraintsMatch(
        _ value: Any,
        schema: JSONObject
    ) -> Bool {
        guard schema["pattern"] is String ||
            schema["minLength"] != nil ||
            schema["maxLength"] != nil else {
            return true
        }
        guard let string = stringValueForConstraintMatching(value) else {
            return false
        }
        if let minimum = integerKeyword("minLength", in: schema),
            string.count < minimum {
            return false
        }
        if let maximum = integerKeyword("maxLength", in: schema),
            string.count > maximum {
            return false
        }
        if let pattern = schema["pattern"] as? String,
            !patternMatches(pattern, propertyName: string) {
            return false
        }
        return true
    }

    static func integerKeyword(
        _ key: String,
        in schema: JSONObject
    ) -> Int? {
        if let value = schema[key] as? Int {
            return value
        }
        if let value = schema[key] as? Double {
            return Int(value)
        }
        if let value = schema[key] as? String {
            return Int(value)
        }
        return nil
    }

    private static func stringValueForConstraintMatching(_ value: Any) -> String? {
        if value is [Any] || value is JSONObject || value is NSNull {
            return nil
        }
        return stringValue(from: value)
    }
}
