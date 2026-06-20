import Foundation

extension ToolSchema {
    static func objectValue(
        from value: Any,
        schema: JSONObject
    ) -> JSONObject? {
        let object = parsedObject(from: value)
        guard let object else {
            return nil
        }
        let schemaWithBranchProperties = schemaWithSelectedCompositionBranches(schema, object: object)
        guard isObjectLike(schemaWithBranchProperties) else {
            return object
        }
        return normalizeNestedObject(object, using: schemaWithBranchProperties)
    }

    static func arrayValue(
        from value: Any,
        schema: JSONObject
    ) -> Any? {
        let values = parsedArray(from: value)
        guard !values.isEmpty || isEmptyArray(value) else {
            return nil
        }
        return normalizedArray(values, using: schema)
    }

    static func stringValue(from value: Any) -> String? {
        if value is NSNull {
            return nil
        }
        if let string = value as? String {
            return string
        }
        if let bool = value as? Bool {
            return bool ? "true" : "false"
        }
        if JSONSerialization.isValidJSONObject(["value": value]) {
            return MLXToolCallParsingSupport.canonicalJSONString(["value": value])
                .droppingJSONValueEnvelope()
        }
        return String(describing: value)
    }

    private static func parsedObject(from value: Any) -> JSONObject? {
        if let parsed = value as? JSONObject {
            return parsed
        }
        guard let string = value as? String else {
            return nil
        }
        return MLXToolCallParsingSupport.parseJSON(string) as? JSONObject
    }

    private static func parsedArray(from value: Any) -> [Any] {
        if let parsed = value as? [Any] {
            return parsed
        }
        guard let string = value as? String else {
            return []
        }
        return MLXToolCallParsingSupport.parseJSON(string) as? [Any] ?? []
    }

    private static func isEmptyArray(_ value: Any) -> Bool {
        if let values = value as? [Any] {
            return values.isEmpty
        }
        guard let string = value as? String else {
            return false
        }
        return string.trimmingCharacters(in: .whitespacesAndNewlines) == "[]"
    }

    static func isObjectLike(_ schema: JSONObject) -> Bool {
        schema["properties"] is JSONObject
            || additionalPropertiesSchema(from: schema) != nil
            || rejectsAdditionalProperties(from: schema)
            || hasPatternProperties(from: schema)
    }

    static func isArrayLike(_ schema: JSONObject) -> Bool {
        directTypes(from: schema["type"]).contains("array")
            || schema["items"] is JSONObject
            || !prefixItemSchemas(from: schema).isEmpty
    }

    private static func normalizedArray(
        _ values: [Any],
        using schema: JSONObject
    ) -> [Any] {
        let prefixSchemas = prefixItemSchemas(from: schema)
        let itemSchema = schema["items"] as? JSONObject
        guard !prefixSchemas.isEmpty || itemSchema != nil else {
            return values
        }
        return values.enumerated().map { index, value in
            if prefixSchemas.indices.contains(index) {
                return normalize(value, using: prefixSchemas[index])
            }
            if let itemSchema {
                return normalize(value, using: itemSchema)
            }
            return value
        }
    }

    private static func prefixItemSchemas(from schema: JSONObject) -> [JSONObject] {
        guard let children = schema["prefixItems"] as? [Any] else {
            return []
        }
        return children.compactMap { $0 as? JSONObject }
    }

    static func schemaWithMergedBranchProperties(_ schema: JSONObject) -> JSONObject {
        guard let branchProperties = branchProperties(from: schema) else {
            return schema
        }
        var merged = schema
        var properties = branchProperties
        if let ownProperties = schema["properties"] as? JSONObject {
            properties.merge(ownProperties) { _, own in own }
        }
        merged["properties"] = properties
        return merged
    }
}
