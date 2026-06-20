import Foundation

extension MLXPromptTemplateRenderer {
    private struct Gemma4RenderState {
        var hasOpenModelTurn = false
    }

    private struct Gemma4RenderContext {
        let messages: [MLXBridgeMessage]
        let style: MLXPromptStyle
        var prompt: String
        var state: Gemma4RenderState
    }

    static func renderGemma4(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var context = Gemma4RenderContext(
            messages: request.messages,
            style: style,
            prompt: "<bos>",
            state: Gemma4RenderState()
        )
        let system = gemma4SystemContent(for: request, style: style)
        if !system.isEmpty {
            context.prompt += gemma4ClosedTurn(
                role: "system",
                content: system,
                state: &context.state
            )
        }

        var index = context.messages.startIndex
        while index < context.messages.endIndex {
            index = appendGemma4Message(at: index, context: &context)
        }

        if !context.state.hasOpenModelTurn {
            context.prompt += "<|turn>model\n"
        }
        context.prompt += reasoningOpenSuffix(for: request, style: style)
        return context.prompt
    }

    private static func appendGemma4Message(
        at index: Int,
        context: inout Gemma4RenderContext
    ) -> Int {
        let message = context.messages[index]
        switch message.role {
        case .assistant:
            return appendGemma4AssistantMessage(at: index, context: &context)

        case .system:
            appendGemma4ClosedRoleMessage(message, role: "system", context: &context)

        case .tool:
            context.prompt += gemma4ToolResponseTurn(
                [message],
                state: &context.state
            )

        case .user:
            appendGemma4ClosedRoleMessage(message, role: "user", context: &context)
        }
        return index + 1
    }

    private static func appendGemma4AssistantMessage(
        at index: Int,
        context: inout Gemma4RenderContext
    ) -> Int {
        let toolMessages = consecutiveToolMessages(in: context.messages, after: index)
        context.prompt += gemma4AssistantTurn(
            context.messages[index],
            toolMessages: toolMessages,
            style: context.style,
            state: &context.state
        )
        return index + 1 + toolMessages.count
    }

    private static func appendGemma4ClosedRoleMessage(
        _ message: MLXBridgeMessage,
        role: String,
        context: inout Gemma4RenderContext
    ) {
        context.prompt += gemma4ClosedTurn(
            role: role,
            content: message.content,
            state: &context.state
        )
    }

    private static func gemma4AssistantTurn(
        _ message: MLXBridgeMessage,
        toolMessages: [MLXBridgeMessage],
        style: MLXPromptStyle,
        state: inout Gemma4RenderState
    ) -> String {
        let content = renderedContent(for: message, style: style)
        let responses = gemma4ToolResponses(toolMessages)
        let shouldContinue = !responses.isEmpty || gemma4ContainsToolCall(content)
        var turn = gemma4ModelPrefix(state: &state) + content + responses
        if shouldContinue {
            if responses.isEmpty {
                turn += "<|tool_response>"
            }
            state.hasOpenModelTurn = true
        } else {
            turn += "<turn|>\n"
            state.hasOpenModelTurn = false
        }
        return turn
    }

    private static func gemma4ClosedTurn(
        role: String,
        content: String,
        state: inout Gemma4RenderState
    ) -> String {
        let prefix: String
        if role == "model" {
            prefix = gemma4ModelPrefix(state: &state)
        } else {
            prefix = gemma4RolePrefix(role, state: &state)
        }
        state.hasOpenModelTurn = false
        return "\(prefix)\(content)<turn|>\n"
    }

    private static func gemma4ToolResponseTurn(
        _ messages: [MLXBridgeMessage],
        state: inout Gemma4RenderState
    ) -> String {
        gemma4ModelPrefix(state: &state) + gemma4ToolResponses(messages)
    }

    private static func gemma4ModelPrefix(state: inout Gemma4RenderState) -> String {
        guard !state.hasOpenModelTurn else {
            return ""
        }
        state.hasOpenModelTurn = true
        return "<|turn>model\n"
    }

    private static func gemma4RolePrefix(
        _ role: String,
        state: inout Gemma4RenderState
    ) -> String {
        let closeOpenTurn = state.hasOpenModelTurn ? "<turn|>\n" : ""
        state.hasOpenModelTurn = false
        return "\(closeOpenTurn)<|turn>\(role)\n"
    }

    private static func gemma4SystemContent(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let sections = systemSections(for: request, style: style, includeTools: false)
        let tools = MLXToolTemplateAdapter.prepare(request.tools, for: .gemma)
        let declarations = tools.sorted { $0.name < $1.name }
            .map(MLXGemma4ValueFormatter.toolDeclaration)
            .joined()
        return (sections + [declarations])
            .filter { !$0.isEmpty }
            .joined(separator: "\n")
    }

    private static func consecutiveToolMessages(
        in messages: [MLXBridgeMessage],
        after index: Int
    ) -> [MLXBridgeMessage] {
        var result: [MLXBridgeMessage] = []
        var cursor = index + 1
        while cursor < messages.endIndex, messages[cursor].role == .tool {
            result.append(messages[cursor])
            cursor += 1
        }
        return result
    }

    private static func gemma4ToolResponses(_ messages: [MLXBridgeMessage]) -> String {
        messages.map { message in
            MLXGemma4ValueFormatter.toolResponse(
                name: message.name ?? "unknown",
                content: message.content
            )
        }
        .joined()
    }

    private static func gemma4ContainsToolCall(_ content: String) -> Bool {
        content.contains("<|tool_call>") || content.contains("<tool_call|>")
    }
}
