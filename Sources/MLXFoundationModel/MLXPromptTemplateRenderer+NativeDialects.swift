import Foundation

extension MLXPromptTemplateRenderer {
    static func renderQwen(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let turns = expandedMessages(for: request, style: style).map { message in
            qwenMessage(message, style: style)
        }
        return (
            turns + [
                "<|im_start|>assistant\n\(reasoningOpenSuffix(for: request, style: style))"
            ]
        )
        .joined(separator: "\n")
    }

    static func renderKimi(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var turns = [kimiToolDeclaration(for: request.tools)]
            .filter { !$0.isEmpty }
        turns.append(contentsOf: expandedMessages(
            for: request,
            style: style,
            includeTools: false
        ).map { message in
            """
            \(kimiRoleStart(for: message))\
            \(kimiContent(for: message, style: style))<|im_end|>
            """
        })
        turns.append("<|im_assistant|>assistant<|im_middle|>")
        return turns.joined(separator: "\n")
    }

    static func renderLongCat(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var prompt = longCatPrelude(for: request, style: style)
        var round = 0
        for message in request.messages {
            prompt += longCatMessage(message, style: style, round: &round)
        }
        return prompt + longCatAssistantContinuation(for: request, style: style)
    }

    static func renderMiniMaxM3(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let developer = systemSections(for: request, style: style)
            .joined(separator: "\n\n")
        var turns = [
            miniMaxM3Turn(
                role: "system",
                content: "You are a helpful assistant.",
                prefix: "]~!b[]~b]"
            ),
            miniMaxM3Turn(role: "developer", content: developer)
        ]
        turns.append(contentsOf: request.messages.map { message in
            miniMaxM3RenderedMessage(message, style: style)
        })
        turns.append("]~b]ai\n\(reasoningOpenSuffix(for: request, style: style))")
        return turns.joined()
    }

    static func renderMiniMaxXML(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var turns = [
            miniMaxXMLTurn(
                role: "system",
                content: miniMaxXMLSystemContent(for: request, style: style),
                prefix: "]~!b[]~b]"
            )
        ]
        turns.append(contentsOf: request.messages.map { message in
            miniMaxXMLRenderedMessage(message, style: style)
        })
        turns.append("]~b]ai\n<think>\n")
        return turns.joined()
    }

    static func renderHarmony(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let body = expandedMessages(for: request, style: style).map { message in
            harmonyMessage(message, style: style)
        }
        return body.joined() + harmonyAssistantContinuation(for: request)
    }

    private static func qwenMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return qwenTurn(role: "assistant", content: renderedContent(for: message, style: style))

        case .system:
            return qwenTurn(role: "system", content: message.content)

        case .tool:
            return qwenTurn(role: "user", content: qwenToolResponseContent(message.content))

        case .user:
            return qwenTurn(role: "user", content: message.content)
        }
    }

    private static func harmonyMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return harmonyAssistantMessage(message, style: style)

        case .system:
            return harmonyFrame(header: "system", content: message.content)

        case .tool:
            return harmonyToolMessage(message)

        case .user:
            return harmonyFrame(header: "user", content: message.content)
        }
    }

    private static func longCatPrelude(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var prompt = ""
        let tools = renderTools(request.tools, style: style)
        if !tools.isEmpty {
            prompt += "\(tools)\n\n## Messages\n"
        }
        let system = systemSections(for: request, style: style, includeTools: false)
            .joined(separator: "\n\n")
        if !system.isEmpty {
            prompt += "SYSTEM:\(system) "
        }
        return prompt
    }

    private static func longCatMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle,
        round: inout Int
    ) -> String {
        switch message.role {
        case .system:
            return "SYSTEM:\(message.content) "

        case .user:
            defer { round += 1 }
            return "[Round \(round)] USER:\(message.content) ASSISTANT:"

        case .assistant:
            return "\(renderedContent(for: message, style: style))</longcat_s> "

        case .tool:
            return "TOOL:\(toolContent(message))</longcat_s> "
        }
    }

    private static func longCatAssistantContinuation(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        guard let lastRole = request.messages.last?.role else {
            return "ASSISTANT:\(reasoningOpenSuffix(for: request, style: style))"
        }
        switch lastRole {
        case .user:
            return reasoningOpenSuffix(for: request, style: style)

        case .tool, .system:
            return "ASSISTANT:\(reasoningOpenSuffix(for: request, style: style))"

        case .assistant:
            return ""
        }
    }

    private static func kimiToolDeclaration(for tools: [MLXBridgeToolDefinition]) -> String {
        guard !tools.isEmpty else {
            return ""
        }
        let tools = MLXToolTemplateAdapter.prepare(tools, for: .kimiK2)
        return """
        <|im_system|>tool_declare<|im_middle|>
        # Tools
        \(canonicalToolsJSON(tools))<|im_end|>
        """
    }

    private static func kimiRoleStart(for message: MLXBridgeMessage) -> String {
        let roleName = message.name ?? chatMLRole(message)
        switch message.role {
        case .user:
            return "<|im_user|>\(roleName)<|im_middle|>"

        case .assistant:
            return "<|im_assistant|>\(roleName)<|im_middle|>"

        case .system, .tool:
            return "<|im_system|>\(roleName)<|im_middle|>"
        }
    }

    private static func kimiContent(
        for message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        if message.role == .tool {
            return "## Return of \(message.name ?? "tool")\n\(message.content)"
        }
        return renderedContent(for: message, style: style)
    }

    private static func miniMaxM3RenderedMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return miniMaxM3Turn(
                role: "ai",
                content: "</mm:think>\(renderedContent(for: message, style: style))"
            )

        case .system:
            return miniMaxM3Turn(role: "developer", content: message.content)

        case .tool:
            return miniMaxM3Turn(role: "tool", content: "<response>\(message.content)</response>")

        case .user:
            return miniMaxM3Turn(role: "user", content: message.content)
        }
    }

    private static func miniMaxXMLSystemContent(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let content = systemSections(for: request, style: style)
            .joined(separator: "\n\n")
        guard !content.isEmpty else {
            return "You are MiniMax-M2, a helpful AI assistant built by MiniMax."
        }
        return content
    }

    private static func miniMaxXMLRenderedMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return miniMaxXMLTurn(role: "ai", content: renderedContent(for: message, style: style))

        case .system:
            return miniMaxXMLTurn(role: "system", content: message.content)

        case .tool:
            return miniMaxXMLTurn(role: "tool", content: "<response>\(message.content)</response>")

        case .user:
            return miniMaxXMLTurn(role: "user", content: message.content)
        }
    }

    private static func harmonyAssistantMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        let calls = MLXToolCallExtractor.extractAll(from: message.content)
        if !calls.isEmpty {
            return MLXToolPromptDialect(style: style).renderToolCalls(calls)
        }
        return harmonyFrame(
            header: "assistant",
            content: renderedContent(for: message, style: style),
            channel: "final"
        )
    }

    private static func harmonyToolMessage(_ message: MLXBridgeMessage) -> String {
        let name = message.name ?? "unknown"
        return harmonyFrame(
            header: "functions.\(name) to=assistant",
            content: harmonyJSONFragment(message.content),
            channel: "commentary"
        )
    }

    private static func harmonyFrame(
        header: String,
        content: String,
        channel: String? = nil,
        terminator: String = "<|end|>"
    ) -> String {
        var frame = "<|start|>\(header)"
        if let channel {
            frame += "<|channel|>\(channel)"
        }
        return "\(frame)<|message|>\(content)\(terminator)"
    }

    private static func harmonyJSONFragment(_ text: String) -> String {
        let value = MLXToolCallParsingSupport.parseJSON(text) ?? text
        if value is MLXToolCallParsingSupport.JSONObject || value is [Any] {
            return MLXToolCallParsingSupport.canonicalJSONString(value)
        }
        return MLXToolCallParsingSupport.canonicalJSONString(["value": value])
            .droppingJSONValueEnvelope()
    }

    private static func qwenTurn(role: String, content: String) -> String {
        "<|im_start|>\(role)\n\(content)<|im_end|>"
    }

    private static func qwenToolResponseContent(_ content: String) -> String {
        """
        <tool_response>
        \(content)
        </tool_response>
        """
    }

    private static func miniMaxM3Turn(
        role: String,
        content: String,
        prefix: String = "]~b]"
    ) -> String {
        "\(prefix)\(role)\n\(content)[e~[\n"
    }

    private static func miniMaxXMLTurn(
        role: String,
        content: String,
        prefix: String = "]~b]"
    ) -> String {
        "\(prefix)\(role)\n\(content)[e~[\n"
    }

    private static func canonicalToolsJSON(_ tools: [MLXBridgeToolDefinition]) -> String {
        let payload = tools.sorted { $0.name < $1.name }.map { tool in
            [
                "function": [
                    "description": tool.description,
                    "name": tool.name,
                    "parameters": MLXToolCallParsingSupport.parseJSON(tool.parametersJSONSchema) ?? [:]
                ],
                "type": "function"
            ] as [String: Any]
        }
        return MLXToolCallParsingSupport.canonicalJSONString(payload)
    }

    private static func harmonyAssistantContinuation(
        for request: MLXBridgeRequest
    ) -> String {
        guard request.effectiveReasoningOptions.isEnabled else {
            return "<|start|>assistant"
        }
        return "<|start|>assistant<|channel|>analysis<|message|>"
    }
}
