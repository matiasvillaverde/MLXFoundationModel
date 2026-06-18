#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsToolSchemaBuilder {
    static func bridgeToolDefinition(
        _ definition: Transcript.ToolDefinition
    ) -> MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: definition.name,
            description: definition.description,
            parametersJSONSchema: FoundationModelsSchemaSupport.jsonSchemaString(from: definition.parameters)
        )
    }

    static func toolCallsText(_ calls: Transcript.ToolCalls) -> String {
        calls.map { call in
            FoundationModelsSchemaSupport.jsonString(from: [
                "arguments": FoundationModelsSchemaSupport.jsonValue(from: call.arguments.jsonString),
                "tool_name": call.toolName
            ])
        }
        .joined(separator: "\n")
    }

    static func requiredToolCallSchema(
        from definitions: [Transcript.ToolDefinition]
    ) -> String {
        let choices = definitions.map(toolCallChoiceSchema)
        let schema = combinedSchema(from: choices)
        return FoundationModelsSchemaSupport.jsonString(from: schema)
    }

    private static func combinedSchema(from choices: [[String: Any]]) -> Any {
        if choices.count == 1 {
            return choices[0]
        }
        return ["oneOf": choices] as [String: Any]
    }

    private static func toolCallChoiceSchema(
        for definition: Transcript.ToolDefinition
    ) -> [String: Any] {
        [
            "additionalProperties": false,
            "properties": [
                "arguments": FoundationModelsSchemaSupport.jsonValue(
                    from: FoundationModelsSchemaSupport.jsonSchemaString(from: definition.parameters)
                ),
                "tool_name": [
                    "enum": [
                        definition.name
                    ]
                ]
            ],
            "required": [
                "tool_name",
                "arguments"
            ],
            "type": "object"
        ]
    }
}
#endif
