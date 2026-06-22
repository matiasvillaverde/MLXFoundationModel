import Foundation

extension MLXPromptTemplateRenderer {
    static func renderApertus(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let system = systemSections(for: request, style: style, includeTools: false)
            .joined(separator: "\n\n")
        let turns = request.messages.map { message in
            apertusMessage(message, style: style)
        }
        return (
            [
                "<s>",
                apertusTurn(role: "system", content: apertusSystemContent(system)),
                apertusDeveloperTurn(for: request, style: style)
            ] + turns + [
                "<|assistant_start|>\(reasoningOpenSuffix(for: request, style: style))"
            ]
        )
        .joined()
    }

    private static func apertusMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return apertusTurn(
                role: "assistant",
                content: renderedContent(for: message, style: style)
            )

        case .system:
            return apertusTurn(role: "system", content: message.content)

        case .tool:
            return apertusTurn(role: "user", content: toolContent(message))

        case .user:
            return apertusTurn(role: "user", content: message.content)
        }
    }

    private static func apertusTurn(role: String, content: String) -> String {
        "<|\(role)_start|>\(content)<|\(role)_end|>"
    }

    private static func apertusDeveloperTurn(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let toolCapabilities = request.tools.isEmpty
            ? "disabled"
            : "\n\(renderTools(request.tools, style: style))"
        let content = """
        Deliberation: disabled
        Tool Capabilities: \(toolCapabilities)
        """
        return apertusTurn(role: "developer", content: content)
    }

    private static func apertusSystemContent(_ content: String) -> String {
        guard !content.isEmpty else {
            return "You are Apertus, a helpful assistant created by the SwissAI initiative."
        }
        return content
    }
}
