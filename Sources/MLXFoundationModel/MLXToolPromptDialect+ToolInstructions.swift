import Foundation

extension MLXToolPromptDialect {
    func toolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        guard !tools.isEmpty else {
            return ""
        }
        let tools = MLXToolTemplateAdapter.prepare(tools, for: self)
        if let instructions = primaryToolInstructions(for: tools) {
            return instructions
        }
        if let instructions = secondaryToolInstructions(for: tools) {
            return instructions
        }
        return genericToolInstructions(for: tools)
    }

    func cohereToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        # Tools
        Tool definitions are available in JSON Schema:

        <tools>
        \(toolPayloads(tools))
        </tools>

        To call a tool, emit a single action object between Cohere action markers:
        \(formatExample(for: tools[0]))
        """
    }

    func deepSeekDSMLToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        ## Tools

        You have access to a set of tools you can use to answer the user's question.
        You can invoke functions by writing a "<\(Self.deepSeekDSMLToken)tool_calls>" block.

        String and scalar parameters should be specified as is without escaping or quotes, \
        while lists and objects should use JSON format. The "string" attribute must be "true" \
        for string parameters and "false" for numbers, booleans, arrays, and objects.

        Here are the functions available in JSONSchema format:
        <functions>
        \(toolPayloads(tools))
        </functions>

        Example:
        \(formatExample(for: tools[0]))
        """
    }

    func gemmaToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        # Tools
        You may call one or more tools. Tool definitions are available in JSON Schema:

        <tools>
        \(toolPayloads(tools))
        </tools>

        To call a tool, use Gemma's function-call format:
        \(formatExample(for: tools[0]))
        """
    }

    func glmXMLToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        # Tools
        Tool definitions are provided in JSON Schema:

        <tools>
        \(toolPayloads(tools))
        </tools>

        To call a tool, emit only GLM XML:
        \(formatExample(for: tools[0]))
        """
    }

    func longCatToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        let toolDescriptions = tools.sorted { $0.name < $1.name }
            .map(longCatToolDescription)
            .joined(separator: "\n\n")

        return """
        ## Tools
        You have access to the following tools:

        ### Tool namespace: function

        \(toolDescriptions)

        **Note**: For each function call, return a JSON object with function name and arguments within \
        <longcat_tool_call></longcat_tool_call> XML tags as follows:
        <longcat_tool_call>
        {"name": <function-name>, "arguments": <args-dict>}
        </longcat_tool_call>
        """
    }

    func harmonyToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        let toolDescriptions = tools.sorted { $0.name < $1.name }
            .map(harmonyToolDescription)
            .joined(separator: "\n\n")

        return """
        # Tools
        You may call tools only from the commentary channel.
        To call a tool, emit a Harmony assistant message addressed to \
        functions.<tool_name> with a JSON object payload, then end it with <|call|>.

        Available function namespace:

        \(toolDescriptions)

        Example:
        \(formatExample(for: tools[0]))
        """
    }

    func miniMaxXMLToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        # Tools
        Tool definitions are provided in JSON Schema:

        <tools>
        \(toolPayloads(tools))
        </tools>

        To call a tool, emit MiniMax invoke XML:
        \(formatExample(for: tools[0]))
        """
    }

    func miniMaxM3ToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        let toolPayload = tools.sorted { $0.name < $1.name }
            .map(miniMaxM3ToolPayload)
            .joined(separator: "\n")

        return """
        # Tools
        You may call one or more tools to assist with the user query.
        Here are the tools available in JSONSchema format:

        <tools>
        \(toolPayload)
        </tools>

        To call tools, wrap all invocations in a single \
        \(Self.miniMaxNamespaceToken)<tool_call>\(Self.miniMaxNamespaceToken)</tool_call> block. \
        Parameter values containing nested objects or arrays are recursively expanded into XML elements.
        Example:
        \(formatExample(for: tools[0]))
        """
    }

    func mistralToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        [AVAILABLE_TOOLS]\(toolArrayPayload(tools))[/AVAILABLE_TOOLS]

        To call a tool, emit Mistral tool-call markup:
        \(formatExample(for: tools[0]))
        """
    }

    func qwenXMLToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        # Tools
        You may call one or more tools. Tool definitions are provided in JSON Schema:

        <tools>
        \(toolPayloads(tools))
        </tools>

        To call a tool, emit only Qwen XML:
        \(formatExample(for: tools[0]))
        """
    }

    func formatExample(for tool: MLXBridgeToolDefinition) -> String {
        let call = MLXExtractedToolCall(
            name: tool.name,
            argumentsJSON: sampleArgumentsJSON(from: tool.parametersJSONSchema)
        )
        return renderToolCall(call, index: 0)
    }

    private func primaryToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String? {
        switch self {
        case .cohereAction:
            return cohereToolInstructions(for: tools)

        case .deepSeekDSML:
            return deepSeekDSMLToolInstructions(for: tools)

        case .functionGemma, .gemma:
            return gemmaToolInstructions(for: tools)

        case .glmXML:
            return glmXMLToolInstructions(for: tools)

        case .qwenXML:
            return qwenXMLToolInstructions(for: tools)

        default:
            return nil
        }
    }

    private func secondaryToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String? {
        switch self {
        case .harmony:
            return harmonyToolInstructions(for: tools)

        case .longCat:
            return longCatToolInstructions(for: tools)

        case .minimaxM3:
            return miniMaxM3ToolInstructions(for: tools)

        case .minimaxXML:
            return miniMaxXMLToolInstructions(for: tools)

        case .mistralToolCall:
            return mistralToolInstructions(for: tools)

        default:
            return nil
        }
    }

    private func genericToolInstructions(for tools: [MLXBridgeToolDefinition]) -> String {
        """
        Available tools:
        \(toolList(tools))

        To call a tool, emit exactly this format:
        \(formatExample(for: tools[0]))
        """
    }

    private func toolList(_ tools: [MLXBridgeToolDefinition]) -> String {
        tools
            .sorted { $0.name < $1.name }
            .map { tool in
                "- \(tool.name): \(tool.description)\n  schema: \(tool.parametersJSONSchema)"
            }
            .joined(separator: "\n")
    }

    private func longCatToolDescription(_ tool: MLXBridgeToolDefinition) -> String {
        """
        #### Tool name: \(tool.name)
        Description: \(tool.description)
        InputSchema:
        \(tool.parametersJSONSchema)
        """
    }

    private func harmonyToolDescription(_ tool: MLXBridgeToolDefinition) -> String {
        """
        ## functions.\(tool.name)
        \(toolPayload(tool))
        """
    }

    private func toolPayloads(_ tools: [MLXBridgeToolDefinition]) -> String {
        tools.sorted { $0.name < $1.name }
            .map { tool in
                "<tool>\(toolPayload(tool))</tool>"
            }
            .joined(separator: "\n")
    }

    private func toolArrayPayload(_ tools: [MLXBridgeToolDefinition]) -> String {
        let payload = tools.sorted { $0.name < $1.name }.map(toolPayloadObject)
        return MLXToolCallParsingSupport.canonicalJSONString(payload)
    }

    private func toolPayload(_ tool: MLXBridgeToolDefinition) -> String {
        MLXToolCallParsingSupport.canonicalJSONString(toolPayloadObject(tool))
    }

    private func toolPayloadObject(_ tool: MLXBridgeToolDefinition) -> [String: Any] {
        [
            "function": [
                "description": tool.description,
                "name": tool.name,
                "parameters": parsedArguments(tool.parametersJSONSchema)
            ],
            "type": "function"
        ]
    }

    private func miniMaxM3ToolPayload(_ tool: MLXBridgeToolDefinition) -> String {
        let payload: [String: Any] = [
            "function": [
                "description": tool.description,
                "name": tool.name,
                "parameters": parsedArguments(tool.parametersJSONSchema)
            ]
        ]
        return "<tool>\(MLXToolCallParsingSupport.canonicalJSONString(payload))</tool>"
    }

    private func sampleArgumentsJSON(from schemaText: String) -> String {
        guard let root = MLXToolCallParsingSupport.parseJSON(schemaText) as? [String: Any] else {
            return "{}"
        }
        let schema = ToolSchema.schemaWithMergedBranchProperties(
            ToolSchema.expandedSchema(root, root: root)
        )
        guard let properties = schema["properties"] as? [String: Any] else {
            return "{}"
        }

        let values = properties.keys.sorted().reduce(into: [String: Any]()) { result, key in
            result[key] = sampleValue(from: properties[key])
        }
        return MLXToolCallParsingSupport.canonicalJSONString(values)
    }

    private func sampleValue(from schema: Any?) -> Any {
        guard let schema = schema as? [String: Any] else {
            return "value"
        }
        if let literal = sampleLiteralValue(from: schema) {
            return literal
        }

        let mergedSchema = ToolSchema.schemaWithMergedBranchProperties(schema)
        switch schemaType(mergedSchema) {
        case "integer":
            return 1

        case "number":
            return 1.0

        case "boolean":
            return true

        case "array":
            return sampleArrayValue(from: mergedSchema)

        case "object":
            return sampleObjectValue(from: mergedSchema)

        default:
            return "value"
        }
    }

    private func sampleLiteralValue(from schema: [String: Any]) -> Any? {
        if let constant = schema["const"] {
            return constant
        }
        if let values = schema["enum"] as? [Any] {
            return values.first
        }
        return nil
    }

    private func sampleArrayValue(from schema: [String: Any]) -> [Any] {
        if let prefixItems = schema["prefixItems"] as? [Any],
            !prefixItems.isEmpty {
            return prefixItems.map(sampleValue)
        }
        return [sampleValue(from: schema["items"])]
    }

    private func sampleObjectValue(from schema: [String: Any]) -> [String: Any] {
        guard let properties = schema["properties"] as? [String: Any] else {
            return [:]
        }
        return properties.keys.sorted().reduce(into: [String: Any]()) { result, key in
            result[key] = sampleValue(from: properties[key])
        }
    }

    private func schemaType(_ schema: [String: Any]) -> String {
        if let type = ToolSchema.schemaTypeOrder(from: schema).first {
            return type
        }
        if ToolSchema.isObjectLike(schema) {
            return "object"
        }
        if ToolSchema.isArrayLike(schema) {
            return "array"
        }
        return "string"
    }
}
