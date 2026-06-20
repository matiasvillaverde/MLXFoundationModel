import Foundation

extension ToolSchema {
    static func expandedSchema(
        _ schema: JSONObject,
        root: JSONObject,
        seenReferences: Set<String> = []
    ) -> JSONObject {
        let referenced = resolvedReference(
            in: schema,
            root: root,
            seenReferences: seenReferences
        )
        var expanded = referenced ?? schema
        expanded.removeValue(forKey: "$ref")

        expandObjectChildren(in: &expanded, root: root, seenReferences: seenReferences)
        expandPropertyChildren(in: &expanded, root: root, seenReferences: seenReferences)
        expandPatternPropertyChildren(in: &expanded, root: root, seenReferences: seenReferences)
        expandDependentSchemaChildren(in: &expanded, root: root, seenReferences: seenReferences)
        expandArrayChildren(in: &expanded, root: root, seenReferences: seenReferences)
        return expanded
    }

    static func branchProperties(from schema: JSONObject) -> JSONObject? {
        var merged: JSONObject = [:]
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            for child in children {
                guard let child = child as? JSONObject else {
                    continue
                }
                if let properties = child["properties"] as? JSONObject {
                    merged.merge(properties) { _, new in new }
                }
                if let nested = branchProperties(from: child) {
                    merged.merge(nested) { _, new in new }
                }
            }
        }
        return merged.isEmpty ? nil : merged
    }

    static func additionalPropertiesSchema(from schema: JSONObject) -> JSONObject? {
        if let child = schema["additionalProperties"] as? JSONObject {
            return child
        }
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            for child in children {
                guard let object = child as? JSONObject,
                    let additionalProperties = additionalPropertiesSchema(from: object) else {
                    continue
                }
                return additionalProperties
            }
        }
        return nil
    }

    static func rejectsAdditionalProperties(from schema: JSONObject) -> Bool {
        if let allowed = schema["additionalProperties"] as? Bool {
            return !allowed
        }
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            for child in children {
                guard let object = child as? JSONObject,
                    rejectsAdditionalProperties(from: object) else {
                    continue
                }
                return true
            }
        }
        return false
    }

    static func hasPatternProperties(from schema: JSONObject) -> Bool {
        if let patterns = schema["patternProperties"] as? JSONObject,
            !patterns.isEmpty {
            return true
        }
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            for child in children {
                guard let object = child as? JSONObject,
                    hasPatternProperties(from: object) else {
                    continue
                }
                return true
            }
        }
        return false
    }

    static func patternPropertySchemas(
        for propertyName: String,
        from schema: JSONObject
    ) -> [JSONObject] {
        var schemas: [JSONObject] = []
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            for child in children {
                guard let object = child as? JSONObject else {
                    continue
                }
                schemas.append(
                    contentsOf: patternPropertySchemas(for: propertyName, from: object)
                )
            }
        }
        guard let patterns = schema["patternProperties"] as? JSONObject else {
            return schemas
        }
        for pattern in patterns.keys.sorted()
            where patternMatches(pattern, propertyName: propertyName) {
            guard let childSchema = patterns[pattern] as? JSONObject else {
                continue
            }
            schemas.append(childSchema)
        }
        return schemas
    }

    private static func expandObjectChildren(
        in schema: inout JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) {
        for key in Self.schemaObjectKeys {
            if let child = schema[key] as? JSONObject {
                schema[key] = expandedSchema(child, root: root, seenReferences: seenReferences)
            }
        }
    }

    private static func expandPropertyChildren(
        in schema: inout JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) {
        guard let properties = schema["properties"] as? JSONObject else {
            return
        }
        schema["properties"] = expandedProperties(
            properties,
            root: root,
            seenReferences: seenReferences
        )
    }

    private static func expandPatternPropertyChildren(
        in schema: inout JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) {
        guard let patterns = schema["patternProperties"] as? JSONObject else {
            return
        }
        schema["patternProperties"] = expandedProperties(
            patterns,
            root: root,
            seenReferences: seenReferences
        )
    }

    private static func expandDependentSchemaChildren(
        in schema: inout JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) {
        guard let dependencies = schema["dependentSchemas"] as? JSONObject else {
            return
        }
        schema["dependentSchemas"] = expandedProperties(
            dependencies,
            root: root,
            seenReferences: seenReferences
        )
    }

    private static func expandArrayChildren(
        in schema: inout JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) {
        for key in Self.schemaArrayKeys {
            guard let children = schema[key] as? [Any] else {
                continue
            }
            schema[key] = children.map { child in
                guard let object = child as? JSONObject else {
                    return child
                }
                return expandedSchema(object, root: root, seenReferences: seenReferences)
            }
        }
    }

    private static let schemaObjectKeys = [
        "additionalProperties",
        "contains",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then"
    ]

    private static let schemaArrayKeys = [
        "allOf",
        "anyOf",
        "oneOf",
        "prefixItems"
    ]

    private static func expandedProperties(
        _ properties: JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) -> JSONObject {
        properties.mapValues { value in
            guard let child = value as? JSONObject else {
                return value
            }
            return expandedSchema(child, root: root, seenReferences: seenReferences)
        }
    }

    private static func resolvedReference(
        in schema: JSONObject,
        root: JSONObject,
        seenReferences: Set<String>
    ) -> JSONObject? {
        guard let reference = schema["$ref"] as? String,
            reference.hasPrefix("#/"),
            !seenReferences.contains(reference),
            let target = value(at: reference, in: root) as? JSONObject
        else {
            return nil
        }

        let expandedTarget = expandedSchema(
            target,
            root: root,
            seenReferences: seenReferences.union([reference])
        )
        var merged = expandedTarget
        for (key, value) in schema where key != "$ref" {
            merged[key] = value
        }
        return merged
    }

    private static func value(at reference: String, in root: JSONObject) -> Any? {
        let path = reference.dropFirst(2).split(separator: "/").map(unescapeReferencePathComponent)
        var current: Any = root
        for component in path {
            if let object = current as? JSONObject {
                guard let next = object[component] else {
                    return nil
                }
                current = next
            } else if let array = current as? [Any],
                let index = Int(component),
                array.indices.contains(index) {
                current = array[index]
            } else {
                return nil
            }
        }
        return current
    }

    private static func unescapeReferencePathComponent(_ component: Substring) -> String {
        component
            .replacingOccurrences(of: "~1", with: "/")
            .replacingOccurrences(of: "~0", with: "~")
    }

    static func patternMatches(_ pattern: String, propertyName: String) -> Bool {
        guard let expression = cachedExpression(for: pattern) else {
            return false
        }
        let range = NSRange(propertyName.startIndex..<propertyName.endIndex, in: propertyName)
        return expression.firstMatch(in: propertyName, options: [], range: range) != nil
    }

    private static func cachedExpression(for pattern: String) -> NSRegularExpression? {
        patternRegexCache.expression(for: pattern)
    }

    private static let patternRegexCache = ToolSchemaPatternRegexCache()
}
