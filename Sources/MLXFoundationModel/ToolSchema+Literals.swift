import Foundation

extension ToolSchema {
    static func canonicalLiteralValue(
        matching value: Any,
        schema: JSONObject
    ) -> Any? {
        if let matched = matchedConst(value, schema: schema) {
            return matched
        }
        if let matched = matchedEnum(value, schema: schema) {
            return matched
        }
        return matchedBranchLiteral(value, schema: schema)
    }

    private static func matchedConst(
        _ value: Any,
        schema: JSONObject
    ) -> Any? {
        guard schema.keys.contains("const"),
            let constValue = schema["const"] else {
            return nil
        }
        return matchedLiteral(value, literal: constValue)
    }

    private static func matchedEnum(
        _ value: Any,
        schema: JSONObject
    ) -> Any? {
        guard let enumValues = schema["enum"] as? [Any] else {
            return nil
        }
        if let exactMatch = enumValues.first(where: { jsonEquivalent(value, $0) }) {
            return exactMatch
        }
        for enumValue in enumValues {
            if let matched = matchedLiteral(value, literal: enumValue) {
                return matched
            }
        }
        return nil
    }

    private static func matchedBranchLiteral(
        _ value: Any,
        schema: JSONObject
    ) -> Any? {
        for key in ["anyOf", "oneOf", "allOf"] {
            guard let branches = schema[key] as? [Any] else {
                continue
            }
            for branch in branches {
                guard let branchSchema = branch as? JSONObject,
                    let matched = canonicalLiteralValue(
                        matching: value,
                        schema: branchSchema
                    ) else {
                    continue
                }
                return matched
            }
        }

        return nil
    }

    private static func matchedLiteral(
        _ value: Any,
        literal: Any
    ) -> Any? {
        if jsonEquivalent(value, literal) {
            return literal
        }
        if literal is NSNull, isNullLiteral(value) {
            return NSNull()
        }

        let kind = literalType(literal)
        if let converted = convertedValue(
            value,
            as: kind,
            schema: ["type": kind]
        ), jsonEquivalent(converted, literal) {
            return literal
        }

        return nil
    }

    private static func jsonEquivalent(
        _ lhs: Any,
        _ rhs: Any
    ) -> Bool {
        if lhs is NSNull, rhs is NSNull {
            return true
        }
        if let lhsBool = lhs as? Bool,
            let rhsBool = rhs as? Bool {
            return lhsBool == rhsBool
        }
        if let lhsString = lhs as? String,
            let rhsString = rhs as? String {
            return lhsString == rhsString
        }
        if let lhsNumber = numericValue(lhs),
            let rhsNumber = numericValue(rhs) {
            return lhsNumber == rhsNumber
        }
        if let lhsArray = lhs as? [Any],
            let rhsArray = rhs as? [Any] {
            return arraysEqual(lhsArray, rhsArray)
        }
        if let lhsObject = lhs as? JSONObject,
            let rhsObject = rhs as? JSONObject {
            return objectsEqual(lhsObject, rhsObject)
        }
        return false
    }

    private static func arraysEqual(
        _ lhs: [Any],
        _ rhs: [Any]
    ) -> Bool {
        guard lhs.count == rhs.count else {
            return false
        }
        return zip(lhs, rhs).allSatisfy { left, right in
            jsonEquivalent(left, right)
        }
    }

    private static func objectsEqual(
        _ lhs: JSONObject,
        _ rhs: JSONObject
    ) -> Bool {
        guard Set(lhs.keys) == Set(rhs.keys) else {
            return false
        }
        for key in lhs.keys {
            guard let left = lhs[key],
                let right = rhs[key],
                jsonEquivalent(left, right) else {
                return false
            }
        }
        return true
    }

    private static func numericValue(_ value: Any) -> Double? {
        if value is Bool {
            return nil
        }
        if let int = value as? Int {
            return Double(int)
        }
        if let double = value as? Double {
            return double
        }
        if let float = value as? Float {
            return Double(float)
        }
        return nil
    }

    private static func literalType(_ value: Any) -> String {
        if value is NSNull {
            return "null"
        }
        if value is Bool {
            return "boolean"
        }
        if value is Int {
            return "integer"
        }
        if value is Double || value is Float {
            return "number"
        }
        if value is [Any] {
            return "array"
        }
        if value is JSONObject {
            return "object"
        }
        return "string"
    }
}
