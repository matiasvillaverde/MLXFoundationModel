import Foundation

struct ToolSchema {
    typealias JSONObject = MLXToolArgumentNormalizer.JSONObject

    private let root: JSONObject
    private let properties: JSONObject

    init(jsonSchema: String) {
        let root = MLXToolCallParsingSupport.parseJSON(jsonSchema) as? JSONObject
        self.root = root.map { Self.expandedSchema($0, root: $0) } ?? [:]
        let mergedRoot = Self.schemaWithMergedBranchProperties(self.root)
        properties = mergedRoot["properties"] as? JSONObject ?? [:]
    }

    func normalizeObject(_ object: JSONObject) -> JSONObject {
        let aliasSchema = Self.schemaWithActiveObjectBranches(root, object: object)
        let aliasProperties = aliasSchema["properties"] as? JSONObject ?? properties
        var normalized = restoreTemplateAliases(in: object, properties: aliasProperties)
        let activeSchema = Self.schemaWithActiveObjectBranches(root, object: normalized)
        let activeProperties = activeSchema["properties"] as? JSONObject ?? [:]
        for (key, value) in object {
            let normalizedKey = restoredAliasName(for: key, properties: activeProperties)
            guard let propertySchema = activeProperties[normalizedKey] as? JSONObject else {
                if let patternValue = Self.normalizedPatternPropertyValue(
                    value,
                    key: normalizedKey,
                    schema: activeSchema
                ) {
                    normalized[normalizedKey] = patternValue
                } else if let additionalSchema = Self.additionalPropertiesSchema(from: activeSchema) {
                    normalized[normalizedKey] = Self.normalize(value, using: additionalSchema)
                } else if Self.rejectsAdditionalProperties(from: activeSchema) {
                    normalized.removeValue(forKey: normalizedKey)
                }
                continue
            }
            normalized[normalizedKey] = Self.normalize(value, using: propertySchema)
        }
        return Self.applyingDefaultProperties(to: normalized, properties: activeProperties)
    }

    private func restoreTemplateAliases(
        in object: JSONObject,
        properties: JSONObject
    ) -> JSONObject {
        var restored = object
        for key in object.keys {
            let restoredKey = restoredAliasName(for: key, properties: properties)
            guard restoredKey != key,
                restored[restoredKey] == nil else {
                continue
            }
            restored[restoredKey] = object[key]
            restored.removeValue(forKey: key)
        }
        return restored
    }

    private func restoredAliasName(for key: String) -> String {
        restoredAliasName(for: key, properties: properties)
    }

    private func restoredAliasName(
        for key: String,
        properties: JSONObject
    ) -> String {
        guard let restored = MLXToolTemplateAdapter.restoredGemmaArgumentName(key),
            properties[restored] != nil,
            properties[key] == nil else {
            return key
        }
        return restored
    }

    static func normalize(
        _ value: Any,
        using schema: JSONObject
    ) -> Any {
        if let literal = canonicalLiteralValue(matching: value, schema: schema) {
            return literal
        }

        let orderedTypes = schemaTypeOrder(from: schema)
        let types = Set(orderedTypes)
        if types.isEmpty {
            return normalizeContainer(value, using: schema)
        }
        if types.contains("null"), isNullLiteral(value) {
            return NSNull()
        }
        for kind in orderedTypes where types.contains(kind) {
            if let converted = convertedValue(value, as: kind, schema: schema) {
                return converted
            }
        }
        return normalizeContainer(value, using: schema)
    }

    static func normalizeNestedObject(
        _ object: JSONObject,
        properties: JSONObject
    ) -> JSONObject {
        normalizeNestedObject(
            object,
            using: [
                "properties": properties,
                "type": "object"
            ]
        )
    }

    static func normalizeNestedObject(
        _ object: JSONObject,
        using schema: JSONObject
    ) -> JSONObject {
        var normalized = object
        let activeSchema = schemaWithActiveObjectBranches(schema, object: object)
        let properties = activeSchema["properties"] as? JSONObject ?? [:]
        let additionalSchema = additionalPropertiesSchema(from: activeSchema)
        for (key, value) in object {
            if let propertySchema = properties[key] as? JSONObject {
                normalized[key] = normalize(value, using: propertySchema)
            } else if let patternValue = normalizedPatternPropertyValue(
                value,
                key: key,
                schema: activeSchema
            ) {
                normalized[key] = patternValue
            } else if let additionalSchema {
                normalized[key] = normalize(value, using: additionalSchema)
            } else if rejectsAdditionalProperties(from: activeSchema) {
                normalized.removeValue(forKey: key)
            }
        }
        return applyingDefaultProperties(to: normalized, properties: properties)
    }

    private static func applyingDefaultProperties(
        to object: JSONObject,
        properties: JSONObject
    ) -> JSONObject {
        var normalized = object
        for key in properties.keys.sorted() where normalized[key] == nil {
            guard let propertySchema = properties[key] as? JSONObject,
                let defaultValue = propertySchema["default"] else {
                continue
            }
            normalized[key] = normalize(defaultValue, using: propertySchema)
        }
        return normalized
    }

    private static func normalizedPatternPropertyValue(
        _ value: Any,
        key: String,
        schema: JSONObject
    ) -> Any? {
        let matchingSchemas = patternPropertySchemas(for: key, from: schema)
        guard !matchingSchemas.isEmpty else {
            return nil
        }
        return matchingSchemas.reduce(value) { current, schema in
            normalize(current, using: schema)
        }
    }
}
