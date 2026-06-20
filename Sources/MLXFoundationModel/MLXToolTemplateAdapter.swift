import Foundation

enum MLXToolTemplateAdapter {
    typealias JSONObject = MLXToolArgumentNormalizer.JSONObject

    private static let gemmaCollidingParameterNames: Set<String> = ["description"]
    private static let gemmaRenamePrefix = "param_"
    private static let schemaChildKeys: Set<String> = [
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then"
    ]
    private static let schemaChoiceKeys: Set<String> = [
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems"
    ]

    static func prepare(
        _ tools: [MLXBridgeToolDefinition],
        for dialect: MLXToolPromptDialect
    ) -> [MLXBridgeToolDefinition] {
        tools.map { tool in
            MLXBridgeToolDefinition(
                name: tool.name,
                description: safeDescription(tool.description),
                parametersJSONSchema: preparedSchema(tool.parametersJSONSchema, dialect: dialect)
            )
        }
    }

    static func restoredGemmaArgumentName(_ name: String) -> String? {
        guard name.hasPrefix(gemmaRenamePrefix) else {
            return nil
        }
        let original = String(name.dropFirst(gemmaRenamePrefix.count))
        return gemmaCollidingParameterNames.contains(original) ? original : nil
    }

    private static func preparedSchema(
        _ schemaText: String,
        dialect: MLXToolPromptDialect
    ) -> String {
        let parsed = MLXToolCallParsingSupport.parseJSON(schemaText) as? JSONObject
        let schema = templateSafeSchema(
            parsed ?? defaultParameterSchema(),
            isSchema: false
        ) as? JSONObject ?? defaultParameterSchema()
        let prepared = shouldEnrichForGemma(dialect)
            ? gemmaEnrichedSchema(schema)
            : schema
        return MLXToolCallParsingSupport.canonicalJSONString(prepared)
    }

    private static func defaultParameterSchema() -> JSONObject {
        [
            "properties": [:] as JSONObject,
            "type": "object"
        ]
    }

    private static func templateSafeSchema(
        _ value: Any,
        isSchema: Bool
    ) -> Any {
        guard let object = value as? JSONObject else {
            return templateSafeValue(value)
        }
        var copied = object.reduce(into: JSONObject()) { result, element in
            result[element.key] = copiedSchemaValue(element.value, key: element.key)
        }
        if isSchema {
            copied["description"] = safeDescription(copied["description"])
        }
        return copied
    }

    private static func copiedSchemaValue(_ value: Any, key: String) -> Any {
        if key == "properties", let properties = value as? JSONObject {
            return copiedProperties(properties)
        }
        if schemaChildKeys.contains(key) {
            return templateSafeSchema(value, isSchema: true)
        }
        if schemaChoiceKeys.contains(key), let choices = value as? [Any] {
            return choices.map { templateSafeSchema($0, isSchema: true) }
        }
        return templateSafeValue(value)
    }

    private static func copiedProperties(_ properties: JSONObject) -> JSONObject {
        properties.reduce(into: JSONObject()) { result, element in
            result[element.key] = templateSafeSchema(element.value, isSchema: true)
        }
    }

    private static func templateSafeValue(_ value: Any) -> Any {
        if let values = value as? [Any] {
            return values.map { templateSafeSchema($0, isSchema: false) }
        }
        return value
    }

    private static func gemmaEnrichedSchema(_ schema: JSONObject) -> JSONObject {
        guard let properties = schema["properties"] as? JSONObject else {
            return schema
        }
        let requiredNames = Set(schema["required"] as? [String] ?? [])
        var enriched = schema
        let mapped = gemmaMappedProperties(properties, requiredNames: requiredNames)
        enriched["properties"] = mapped.properties
        if let required = schema["required"] as? [String] {
            enriched["required"] = required.map { mapped.requiredNameMap[$0] ?? $0 }
        }
        return enriched
    }

    private static func gemmaMappedProperties(
        _ properties: JSONObject,
        requiredNames: Set<String>
    ) -> (properties: JSONObject, requiredNameMap: [String: String]) {
        let existingNames = Set(properties.keys)
        var mappedProperties = JSONObject()
        var requiredNameMap: [String: String] = [:]
        for key in properties.keys.sorted() {
            let mappedKey = gemmaSchemaParameterName(key, existingNames: existingNames)
            mappedProperties[mappedKey] = gemmaEnrichedProperty(
                properties[key],
                originalName: key,
                isRequired: requiredNames.contains(key)
            )
            requiredNameMap[key] = mappedKey
        }
        return (mappedProperties, requiredNameMap)
    }

    private static func gemmaEnrichedProperty(
        _ value: Any?,
        originalName: String,
        isRequired: Bool
    ) -> Any {
        guard var property = value as? JSONObject else {
            return value ?? [:] as JSONObject
        }
        if safeDescription(property["description"]).isEmpty {
            let prefix = isRequired ? "REQUIRED. " : ""
            let typeText = schemaTypeText(property)
            property["description"] = "\(prefix)The '\(originalName)' value (type: \(typeText))"
        }
        return property
    }

    private static func gemmaSchemaParameterName(
        _ name: String,
        existingNames: Set<String>
    ) -> String {
        guard gemmaCollidingParameterNames.contains(name) else {
            return name
        }
        let candidate = "\(gemmaRenamePrefix)\(name)"
        return existingNames.contains(candidate) ? name : candidate
    }

    private static func safeDescription(_ value: Any?) -> String {
        guard let value, !(value is NSNull) else {
            return ""
        }
        return value as? String ?? String(describing: value)
    }

    private static func shouldEnrichForGemma(_ dialect: MLXToolPromptDialect) -> Bool {
        if case .gemma = dialect {
            return true
        }
        return false
    }

    private static func schemaTypeText(_ schema: JSONObject) -> String {
        if let type = schema["type"] as? String {
            return type
        }
        if let types = schema["type"] as? [String] {
            return types.joined(separator: "|")
        }
        return "string"
    }
}
