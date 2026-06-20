#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
extension FMRequiredToolGrammarBuilder {
    static func structuralTagGrammar(
        from definitions: [Transcript.ToolDefinition],
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
        definition: Transcript.ToolDefinition,
        format: FMNativeToolGrammarFormat,
        style: String
    ) -> [String: Any] {
        [
            "type": "tag",
            "begin": format.structuralTagBegin(toolName: definition.name),
            "content": [
                "type": "json_schema",
                "style": style,
                "json_schema": structuralSchemaValue(for: definition)
            ],
            "end": format.structuralTagEnd
        ]
    }

    private static func structuralSchemaValue(
        for definition: Transcript.ToolDefinition
    ) -> Any {
        let schema = FoundationModelsSchemaSupport.jsonSchemaString(from: definition.parameters)
        return FoundationModelsSchemaSupport.jsonValue(from: schema)
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
#endif
