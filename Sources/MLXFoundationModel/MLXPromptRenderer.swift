import Foundation
import MLXLocalModels

/// Converts Foundation Models-style transcript data into text for local MLX models.
public enum MLXPromptRenderer {
    /// Render a bridge request into a prompt string and cache metadata.
    public static func render(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> MLXRenderedRequest {
        let rendererID = "mlx.\(style.codingValue).v1"
        let prompt: String
        switch style {
        case .plain:
            prompt = renderPlain(request)

        case .chatML:
            prompt = renderChatML(request)
        }
        let fingerprint = PromptCacheIdentity.stableFingerprint(for: "\(rendererID)\n\(prompt)")
        return MLXRenderedRequest(
            prompt: prompt,
            rendererID: rendererID,
            cacheFingerprint: fingerprint
        )
    }

    private static func renderPlain(_ request: MLXBridgeRequest) -> String {
        var sections: [String] = []
        if let instructions = request.instructions, !instructions.isEmpty {
            sections.append("System:\n\(instructions)")
        }
        let toolText = renderTools(request.tools)
        if !toolText.isEmpty {
            sections.append(toolText)
        }
        sections.append(contentsOf: request.messages.map(renderPlainMessage))
        sections.append("Assistant:")
        return sections.joined(separator: "\n\n")
    }

    private static func renderPlainMessage(_ message: MLXBridgeMessage) -> String {
        switch message.role {
        case .system:
            return "System:\n\(message.content)"

        case .user:
            return "User:\n\(message.content)"

        case .assistant:
            return "Assistant:\n\(message.content)"

        case .tool:
            let name = message.name.map { " \($0)" } ?? ""
            return "Tool\(name):\n\(message.content)"
        }
    }

    private static func renderChatML(_ request: MLXBridgeRequest) -> String {
        var messages: [MLXBridgeMessage] = []
        let systemContent = [request.instructions, renderTools(request.tools)]
            .compactMap(\.self)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        if !systemContent.isEmpty {
            messages.append(MLXBridgeMessage(role: .system, content: systemContent))
        }
        messages.append(contentsOf: request.messages)

        let rendered = messages.map { message in
            "<|im_start|>\(chatMLRole(message))\n\(message.content)<|im_end|>"
        }
        return (rendered + ["<|im_start|>assistant\n"]).joined(separator: "\n")
    }

    private static func chatMLRole(_ message: MLXBridgeMessage) -> String {
        switch message.role {
        case .system:
            return "system"

        case .user:
            return "user"

        case .assistant:
            return "assistant"

        case .tool:
            return "tool"
        }
    }

    private static func renderTools(_ tools: [MLXBridgeToolDefinition]) -> String {
        guard !tools.isEmpty else {
            return ""
        }
        let definitions = tools
            .sorted { $0.name < $1.name }
            .map { tool in
                "- \(tool.name): \(tool.description)\n  schema: \(tool.parametersJSONSchema)"
            }
            .joined(separator: "\n")
        return """
        Available tools:
        \(definitions)

        To call a tool, respond with a single JSON object:
        {"tool_name":"tool_name","arguments":{}}
        """
    }
}
