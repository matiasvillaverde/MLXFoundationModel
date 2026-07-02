import Foundation
import MLXLocalModels

extension MLXRequiredToolGrammarBuilder {
    static func structuralTagGrammar(
        from definitions: [MLXBridgeToolDefinition],
        format: FMNativeToolGrammarFormat
    ) -> GrammarSamplingConfiguration? {
        guard
            let style = format.structuralTagSchemaStyle,
            !definitions.isEmpty
        else {
            return nil
        }
        let tags = definitions.map { definition in
            structuralToolTag(definition: definition, format: format, style: style)
        }
        let root: [String: Any] = [
            "type": "structural_tag",
            "format": structuralFormat(from: tags)
        ]
        return serializedJSON(root).map(GrammarSamplingConfiguration.structuralTag)
    }

    private static func structuralFormat(
        from tags: [[String: Any]]
    ) -> [String: Any] {
        guard tags.count != 1 else {
            return tags[0]
        }
        return [
            "type": "or",
            "elements": tags
        ]
    }

    private static func structuralToolTag(
        definition: MLXBridgeToolDefinition,
        format: FMNativeToolGrammarFormat,
        style: String
    ) -> [String: Any] {
        [
            "type": "tag",
            "begin": format.structuralTagBegin(toolName: definition.name),
            "content": [
                "type": "json_schema",
                "style": style,
                "json_schema": schemaValue(for: definition)
            ],
            "end": format.structuralTagEnd
        ]
    }

    private static func serializedJSON(_ object: [String: Any]) -> String? {
        guard
            JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(
                withJSONObject: object,
                options: [.sortedKeys]
            )
        else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }
}
