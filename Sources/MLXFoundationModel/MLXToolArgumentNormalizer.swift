import Foundation

enum MLXToolArgumentNormalizer {
    typealias JSONObject = [String: Any]

    static func normalize(
        _ calls: [MLXExtractedToolCall],
        using tools: [MLXBridgeToolDefinition]
    ) -> [MLXExtractedToolCall] {
        guard !calls.isEmpty, !tools.isEmpty else {
            return calls
        }
        let schemas = Dictionary(uniqueKeysWithValues: tools.map { tool in
            (tool.name, ToolSchema(jsonSchema: tool.parametersJSONSchema))
        })
        let validNames = Set(schemas.keys)
        return calls.map { call in
            let name = remappedName(for: call.name, validNames: validNames)
            let remappedCall = name == call.name
                ? call
                : MLXExtractedToolCall(name: name, argumentsJSON: call.argumentsJSON)
            guard let schema = schemas[name] else {
                return remappedCall
            }
            return normalize(remappedCall, using: schema)
        }
    }

    private static func remappedName(
        for emittedName: String,
        validNames: Set<String>
    ) -> String {
        guard !validNames.contains(emittedName),
            emittedName.contains(":") else {
            return emittedName
        }

        let parts = emittedName.split(separator: ":", omittingEmptySubsequences: false)
        guard parts.count > 1 else {
            return emittedName
        }

        let candidates = Set((1..<parts.count).compactMap { index -> String? in
            let suffix = parts[index...].joined(separator: ":")
            return validNames.contains(suffix) ? suffix : nil
        })
        if candidates.count == 1, let candidate = candidates.first {
            return candidate
        }
        return emittedName
    }

    private static func normalize(
        _ call: MLXExtractedToolCall,
        using schema: ToolSchema
    ) -> MLXExtractedToolCall {
        guard let arguments = MLXToolCallParsingSupport.parseJSON(call.argumentsJSON) as? JSONObject else {
            return call
        }
        return MLXExtractedToolCall(
            name: call.name,
            argumentsJSON: MLXToolCallParsingSupport.canonicalJSONString(
                schema.normalizeObject(arguments)
            )
        )
    }
}
