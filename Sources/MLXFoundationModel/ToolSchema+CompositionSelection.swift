import Foundation

extension ToolSchema {
    private struct ObjectBranchCandidate {
        let schema: JSONObject
        let index: Int
        let score: Int
        let exactMatch: Bool
    }

    static func schemaWithSelectedCompositionBranches(
        _ schema: JSONObject,
        object: JSONObject
    ) -> JSONObject {
        var merged = schema
        merged.removeValue(forKey: "allOf")
        merged.removeValue(forKey: "anyOf")
        merged.removeValue(forKey: "oneOf")

        for branch in compositionBranches(from: schema["allOf"]) {
            merged = mergedObjectSchema(merged, with: selectedCompositionSchema(branch, object: object))
        }

        if let branch = selectedExclusiveBranch(
            from: compositionBranches(from: schema["oneOf"]),
            object: object
        ) {
            merged = mergedObjectSchema(merged, with: selectedCompositionSchema(branch, object: object))
        }

        for branch in selectedInclusiveBranches(
            from: compositionBranches(from: schema["anyOf"]),
            object: object
        ) {
            merged = mergedObjectSchema(merged, with: selectedCompositionSchema(branch, object: object))
        }

        return merged
    }

    static func mergedObjectSchema(
        _ base: JSONObject,
        with branch: JSONObject
    ) -> JSONObject {
        var merged = base
        mergeObjectMap("properties", from: branch, into: &merged)
        mergeObjectMap("patternProperties", from: branch, into: &merged)
        mergeObjectMap("dependentSchemas", from: branch, into: &merged)
        if branch.keys.contains("additionalProperties") {
            merged["additionalProperties"] = branch["additionalProperties"]
        }
        return merged
    }

    private static func selectedCompositionSchema(
        _ branch: JSONObject,
        object: JSONObject
    ) -> JSONObject {
        schemaWithSelectedCompositionBranches(branch, object: object)
    }

    private static func selectedExclusiveBranch(
        from branches: [JSONObject],
        object: JSONObject
    ) -> JSONObject? {
        let candidates = objectBranchCandidates(from: branches, object: object)
        let exactMatches = candidates.filter(\.exactMatch)
        if exactMatches.count == 1 {
            return exactMatches[0].schema
        }
        return highestScoringCandidate(from: candidates)?.schema
    }

    private static func selectedInclusiveBranches(
        from branches: [JSONObject],
        object: JSONObject
    ) -> [JSONObject] {
        let candidates = objectBranchCandidates(from: branches, object: object)
        let exactMatches = candidates.filter(\.exactMatch)
        if !exactMatches.isEmpty {
            return exactMatches.map(\.schema)
        }
        return highestScoringCandidate(from: candidates).map { [$0.schema] } ?? []
    }

    private static func objectBranchCandidates(
        from branches: [JSONObject],
        object: JSONObject
    ) -> [ObjectBranchCandidate] {
        branches.enumerated().map { index, branch in
            let exactMatch = objectMatchesSchema(object, schema: branch)
            return ObjectBranchCandidate(
                schema: branch,
                index: index,
                score: objectBranchScore(branch, object: object, exactMatch: exactMatch),
                exactMatch: exactMatch
            )
        }
    }

    private static func highestScoringCandidate(
        from candidates: [ObjectBranchCandidate]
    ) -> ObjectBranchCandidate? {
        candidates.filter { $0.score > 0 }.max { left, right in
            if left.score != right.score {
                return left.score < right.score
            }
            return left.index > right.index
        }
    }

    private static func objectBranchScore(
        _ branch: JSONObject,
        object: JSONObject,
        exactMatch: Bool
    ) -> Int {
        let baseScore = exactMatch ? 100 : 0
        let propertyScore = object.keys.sorted().reduce(0) { score, key in
            guard let value = object[key] else {
                return score
            }
            return score + branchScore(for: value, key: key, branch: branch)
        }
        return baseScore + propertyScore + requiredPropertyScore(branch, object: object)
    }

    private static func branchScore(
        for value: Any,
        key: String,
        branch: JSONObject
    ) -> Int {
        let properties = branch["properties"] as? JSONObject ?? [:]
        if let propertySchema = properties[key] as? JSONObject {
            return matchedPropertyScore(value, schema: propertySchema)
        }
        if !patternPropertySchemas(for: key, from: branch).isEmpty {
            return 1
        }
        if additionalPropertiesSchema(from: branch) != nil {
            return 1
        }
        return rejectsAdditionalProperties(from: branch) ? -3 : 0
    }

    private static func matchedPropertyScore(
        _ value: Any,
        schema: JSONObject
    ) -> Int {
        var score = valueMatchesSchema(value, schema: schema) ? 8 : 2
        if canonicalLiteralValue(matching: value, schema: schema) != nil {
            score += 20
        } else if schema["const"] != nil || schema["enum"] != nil {
            score -= 8
        }
        return score
    }

    private static func requiredPropertyScore(
        _ branch: JSONObject,
        object: JSONObject
    ) -> Int {
        guard let required = branch["required"] as? [String] else {
            return 0
        }
        return required.reduce(0) { score, key in
            score + (object.keys.contains(key) ? 4 : -12)
        }
    }

    private static func compositionBranches(from value: Any?) -> [JSONObject] {
        guard let children = value as? [Any] else {
            return []
        }
        return children.compactMap { $0 as? JSONObject }
    }

    private static func mergeObjectMap(
        _ key: String,
        from source: JSONObject,
        into target: inout JSONObject
    ) {
        guard let sourceMap = source[key] as? JSONObject else {
            return
        }
        var targetMap = target[key] as? JSONObject ?? [:]
        targetMap.merge(sourceMap) { _, new in new }
        target[key] = targetMap
    }
}
