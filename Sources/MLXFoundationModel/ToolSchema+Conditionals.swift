import Foundation

extension ToolSchema {
    private struct ArrayMatchCandidate {
        let values: [Any]
    }

    static func schemaWithActiveObjectBranches(
        _ schema: JSONObject,
        object: JSONObject
    ) -> JSONObject {
        schemaWithActiveObjectBranches(schema, object: object, depth: 0)
    }

    private static func schemaWithActiveObjectBranches(
        _ schema: JSONObject,
        object: JSONObject,
        depth: Int
    ) -> JSONObject {
        guard depth < 8 else {
            return schemaWithMergedBranchProperties(schema)
        }

        var merged = schemaWithSelectedCompositionBranches(schema, object: object)
        for branch in activeObjectBranchSchemas(from: schema, object: object) {
            let expandedBranch = schemaWithActiveObjectBranches(
                branch,
                object: object,
                depth: depth + 1
            )
            merged = mergedObjectSchema(merged, with: expandedBranch)
        }
        return merged
    }

    static func objectMatchesSchema(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        guard compositionMatches(object, schema: schema) else {
            return false
        }
        if let required = schema["required"] as? [String],
            !required.allSatisfy({ object.keys.contains($0) }) {
            return false
        }
        guard dependentRequiredMatches(object, schema: schema) else {
            return false
        }
        guard propertyNamesMatch(object, schema: schema),
            objectPropertyValuesMatch(object, schema: schema),
            objectPatternPropertyValuesMatch(object, schema: schema),
            additionalPropertiesMatch(object, schema: schema) else {
            return false
        }
        return valueTypeMatches(object, schema: schema)
    }

    private static func dependentRequiredMatches(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        guard let dependencies = schema["dependentRequired"] as? JSONObject else {
            return true
        }
        for key in dependencies.keys.sorted() where object.keys.contains(key) {
            let required = stringArray(from: dependencies[key])
            guard required.allSatisfy({ object.keys.contains($0) }) else {
                return false
            }
        }
        return true
    }

    private static func stringArray(from value: Any?) -> [String] {
        if let values = value as? [String] {
            return values
        }
        if let values = value as? [Any] {
            return values.compactMap { $0 as? String }
        }
        if let value = value as? String {
            return [value]
        }
        return []
    }

    private static func propertyNamesMatch(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        guard let propertyNameSchema = schema["propertyNames"] as? JSONObject else {
            return true
        }
        return object.keys.allSatisfy { key in
            valueMatchesSchema(key, schema: propertyNameSchema)
        }
    }

    private static func activeObjectBranchSchemas(
        from schema: JSONObject,
        object: JSONObject
    ) -> [JSONObject] {
        var branches: [JSONObject] = []
        branches.append(contentsOf: conditionalBranches(from: schema, object: object))
        branches.append(contentsOf: dependentBranches(from: schema, object: object))
        return branches
    }

    private static func conditionalBranches(
        from schema: JSONObject,
        object: JSONObject
    ) -> [JSONObject] {
        guard let condition = schema["if"] as? JSONObject else {
            return []
        }
        if objectMatchesSchema(object, schema: condition) {
            return (schema["then"] as? JSONObject).map { [$0] } ?? []
        }
        return (schema["else"] as? JSONObject).map { [$0] } ?? []
    }

    private static func dependentBranches(
        from schema: JSONObject,
        object: JSONObject
    ) -> [JSONObject] {
        guard let dependencies = schema["dependentSchemas"] as? JSONObject else {
            return []
        }
        return dependencies.keys.sorted().compactMap { key in
            guard object.keys.contains(key) else {
                return nil
            }
            return dependencies[key] as? JSONObject
        }
    }

    private static func compositionMatches(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        if let allOf = schema["allOf"] as? [Any],
            !allOf.compactMap({ $0 as? JSONObject }).allSatisfy({ branch in
                objectMatchesSchema(object, schema: branch)
            }) {
            return false
        }
        if let anyOf = schema["anyOf"] as? [Any] {
            let candidates = anyOf.compactMap { $0 as? JSONObject }
            if !candidates.isEmpty,
                !candidates.contains(where: { objectMatchesSchema(object, schema: $0) }) {
                return false
            }
        }
        if let oneOf = schema["oneOf"] as? [Any] {
            let matches = oneOf.compactMap { $0 as? JSONObject }
                .filter { objectMatchesSchema(object, schema: $0) }
            if matches.count != 1 {
                return false
            }
        }
        if let notSchema = schema["not"] as? JSONObject,
            objectMatchesSchema(object, schema: notSchema) {
            return false
        }
        return true
    }

    private static func objectPropertyValuesMatch(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        guard let properties = schema["properties"] as? JSONObject else {
            return true
        }
        for (key, propertySchema) in properties {
            guard let value = object[key],
                let propertySchema = propertySchema as? JSONObject else {
                continue
            }
            guard valueMatchesSchema(value, schema: propertySchema) else {
                return false
            }
        }
        return true
    }

    private static func objectPatternPropertyValuesMatch(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        for (key, value) in object {
            for patternSchema in patternPropertySchemas(for: key, from: schema)
                where !valueMatchesSchema(value, schema: patternSchema) {
                return false
            }
        }
        return true
    }

    private static func additionalPropertiesMatch(
        _ object: JSONObject,
        schema: JSONObject
    ) -> Bool {
        let propertyNames = Set((schema["properties"] as? JSONObject ?? [:]).keys)
        if rejectsAdditionalProperties(from: schema) {
            for key in object.keys where !propertyNames.contains(key) {
                guard patternPropertySchemas(for: key, from: schema).isEmpty else {
                    continue
                }
                return false
            }
        }
        guard let additionalSchema = additionalPropertiesSchema(from: schema) else {
            return true
        }
        for (key, value) in object where !propertyNames.contains(key) {
            guard patternPropertySchemas(for: key, from: schema).isEmpty else {
                continue
            }
            guard valueMatchesSchema(value, schema: additionalSchema) else {
                return false
            }
        }
        return true
    }

    static func valueMatchesSchema(
        _ value: Any,
        schema: JSONObject
    ) -> Bool {
        if canonicalLiteralValue(matching: value, schema: schema) != nil {
            return true
        }
        if schema.keys.contains("const") || schema["enum"] is [Any] {
            return false
        }
        guard valueCompositionMatches(value, schema: schema),
            valueTypeMatches(value, schema: schema),
            arrayContainsMatches(value, schema: schema),
            stringConstraintsMatch(value, schema: schema) else {
            return false
        }
        if let object = objectValueForMatching(value) {
            return objectMatchesSchema(object, schema: schema)
        }
        return true
    }

    private static func valueCompositionMatches(
        _ value: Any,
        schema: JSONObject
    ) -> Bool {
        if let allOf = schema["allOf"] as? [Any],
            !allOf.compactMap({ $0 as? JSONObject }).allSatisfy({ branch in
                valueMatchesSchema(value, schema: branch)
            }) {
            return false
        }
        if let anyOf = schema["anyOf"] as? [Any] {
            let candidates = anyOf.compactMap { $0 as? JSONObject }
            if !candidates.isEmpty,
                !candidates.contains(where: { valueMatchesSchema(value, schema: $0) }) {
                return false
            }
        }
        if let oneOf = schema["oneOf"] as? [Any] {
            let matches = oneOf.compactMap { $0 as? JSONObject }
                .filter { valueMatchesSchema(value, schema: $0) }
            if matches.count != 1 {
                return false
            }
        }
        if let notSchema = schema["not"] as? JSONObject,
            valueMatchesSchema(value, schema: notSchema) {
            return false
        }
        return true
    }

    private static func arrayContainsMatches(
        _ value: Any,
        schema: JSONObject
    ) -> Bool {
        guard let containsSchema = schema["contains"] as? JSONObject else {
            return true
        }
        guard let candidate = arrayValueForMatching(value) else {
            return false
        }
        let matchCount = candidate.values.reduce(into: 0) { count, element in
            if valueMatchesSchema(element, schema: containsSchema) {
                count += 1
            }
        }
        let minimum = integerKeyword("minContains", in: schema) ?? 1
        guard matchCount >= minimum else {
            return false
        }
        if let maximum = integerKeyword("maxContains", in: schema),
            matchCount > maximum {
            return false
        }
        return true
    }

    private static func valueTypeMatches(
        _ value: Any,
        schema: JSONObject
    ) -> Bool {
        let types = schemaTypeOrder(from: schema)
        guard !types.isEmpty else {
            return true
        }
        return types.contains { type in valueMatchesType(value, type: type) }
    }

    private static func valueMatchesType(
        _ value: Any,
        type: String
    ) -> Bool {
        let matchers: [String: (Any) -> Bool] = [
            "null": { candidate in isNullLiteral(candidate) },
            "integer": { candidate in integerValue(from: candidate) != nil },
            "number": { candidate in numberValue(from: candidate) != nil },
            "boolean": { candidate in booleanValue(from: candidate) != nil },
            "object": { candidate in objectValueForMatching(candidate) != nil },
            "array": { candidate in valueIsArrayForMatching(candidate) },
            "string": { candidate in stringValueForMatching(candidate) != nil }
        ]
        return matchers[type]?(value) ?? true
    }

    private static func objectValueForMatching(_ value: Any) -> JSONObject? {
        if let object = value as? JSONObject {
            return object
        }
        guard let string = value as? String else {
            return nil
        }
        return MLXToolCallParsingSupport.parseJSON(string) as? JSONObject
    }

    private static func valueIsArrayForMatching(_ value: Any) -> Bool {
        arrayValueForMatching(value) != nil
    }

    private static func arrayValueForMatching(_ value: Any) -> ArrayMatchCandidate? {
        if let values = value as? [Any] {
            return ArrayMatchCandidate(values: values)
        }
        guard let string = value as? String else {
            return nil
        }
        guard let values = MLXToolCallParsingSupport.parseJSON(string) as? [Any] else {
            return nil
        }
        return ArrayMatchCandidate(values: values)
    }

    private static func stringValueForMatching(_ value: Any) -> String? {
        if value is [Any] || value is JSONObject || value is NSNull {
            return nil
        }
        return stringValue(from: value)
    }
}
