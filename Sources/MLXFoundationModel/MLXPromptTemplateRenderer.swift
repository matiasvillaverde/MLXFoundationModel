import Foundation
import MLXLocalModels

enum MLXPromptTemplateRenderer {
    private typealias RenderingStrategy = @Sendable (MLXBridgeRequest, MLXPromptStyle) -> String

    private static let strategies: [MLXPromptStyle: RenderingStrategy] = [
        .chatML: { Self.renderChatML($0, style: $1) },
        .cohereAction: { Self.renderCohere($0, style: $1) },
        .deepSeekDSML: { Self.renderDeepSeekDSML($0, style: $1) },
        .functionGemma: { Self.renderGemma($0, style: $1) },
        .gemma: { Self.renderGemma4($0, style: $1) },
        .glmXML: { Self.renderGLM($0, style: $1) },
        .harmony: { Self.renderHarmony($0, style: $1) },
        .kimiK2: { Self.renderKimi($0, style: $1) },
        .longCat: { Self.renderLongCat($0, style: $1) },
        .minimaxM3: { Self.renderMiniMaxM3($0, style: $1) },
        .minimaxXML: { Self.renderMiniMaxXML($0, style: $1) },
        .mistralToolCall: { Self.renderMistral($0, style: $1) },
        .plain: { Self.renderPlain($0, style: $1) },
        .qwenXML: { Self.renderQwen($0, style: $1) }
    ]

    static func render(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        (strategies[style] ?? Self.renderPlain)(request, style)
    }

    private static func renderPlain(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var sections: [String] = []
        let toolText = renderTools(request.tools, style: style)
        if !toolText.isEmpty {
            sections.append(toolText)
        }
        if let instructions = request.instructions, !instructions.isEmpty {
            sections.append("System:\n\(instructions)")
        }
        if let constraintText = renderResponseConstraint(request.responseConstraint) {
            sections.append(constraintText)
        }
        sections.append(contentsOf: request.messages.map { message in
            renderPlainMessage(message, style: style)
        })
        sections.append("Assistant:\(reasoningOpenSuffix(for: request, style: style))")
        return sections.joined(separator: "\n\n")
    }

    private static func renderChatML(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let rendered = expandedMessages(for: request, style: style).map { message in
            let content = renderedContent(for: message, style: style)
            return "<|im_start|>\(chatMLRole(message))\n\(content)<|im_end|>"
        }
        return (
            rendered + [
                "<|im_start|>assistant\n\(reasoningOpenSuffix(for: request, style: style))"
            ]
        )
        .joined(separator: "\n")
    }

    private static func renderGLM(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let body = expandedMessages(for: request, style: style).map { message in
            "<|\(glmRole(message))|>\n\(renderedContent(for: message, style: style))"
        }
        return "[gMASK]<sop>\(body.joined())<|assistant|>\(reasoningOpenSuffix(for: request, style: style))"
    }

    private static func renderGemma(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var pendingSystem = ""
        var turns: [String] = []
        for message in expandedMessages(for: request, style: style) {
            appendGemmaTurn(
                message,
                style: style,
                pendingSystem: &pendingSystem,
                turns: &turns
            )
        }
        if !pendingSystem.isEmpty {
            turns.append(gemmaTurn(role: "user", content: pendingSystem))
        }
        turns.append("<start_of_turn>model\n")
        return turns.joined(separator: "\n") + reasoningOpenSuffix(for: request, style: style)
    }

    private static func renderMistral(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        expandedMessages(for: request, style: style).map { message in
            switch message.role {
            case .system:
                return "[SYSTEM_PROMPT]\(message.content)[/SYSTEM_PROMPT]"

            case .user:
                return "[INST]\(message.content)[/INST]"

            case .assistant:
                return renderedContent(for: message, style: style)

            case .tool:
                return "[TOOL_RESULTS]\(toolContent(message))[/TOOL_RESULTS]"
            }
        }
        .joined()
    }

    private static func renderCohere(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let turns = expandedMessages(for: request, style: style).map { message in
            cohereTurn(role: cohereRole(message), content: renderedContent(for: message, style: style))
        }
        return turns.joined()
            + "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
            + reasoningOpenSuffix(for: request, style: style)
    }

    private static func appendGemmaTurn(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle,
        pendingSystem: inout String,
        turns: inout [String]
    ) {
        switch message.role {
        case .system:
            pendingSystem = joinedNonEmpty([pendingSystem, message.content])

        case .user:
            let content = joinedNonEmpty([pendingSystem, message.content])
            pendingSystem = ""
            turns.append(gemmaTurn(role: "user", content: content))

        case .assistant:
            turns.append(gemmaTurn(
                role: "model",
                content: renderedContent(for: message, style: style)
            ))

        case .tool:
            turns.append(gemmaTurn(role: "user", content: toolContent(message)))
        }
    }

    static func expandedMessages(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle,
        includeTools: Bool = true
    ) -> [MLXBridgeMessage] {
        let content = systemSections(for: request, style: style, includeTools: includeTools)
            .joined(separator: "\n\n")
        guard !content.isEmpty else {
            return request.messages
        }
        return [MLXBridgeMessage(role: .system, content: content)] + request.messages
    }

    static func systemSections(
        for request: MLXBridgeRequest,
        style: MLXPromptStyle,
        includeTools: Bool = true
    ) -> [String] {
        [
            includeTools ? renderTools(request.tools, style: style) : nil,
            request.instructions,
            renderResponseConstraint(request.responseConstraint)
        ]
        .compactMap(\.self)
        .filter { !$0.isEmpty }
    }

    private static func renderPlainMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .system:
            return "System:\n\(message.content)"

        case .user:
            return "User:\n\(message.content)"

        case .assistant:
            return "Assistant:\n\(renderedContent(for: message, style: style))"

        case .tool:
            let name = message.name.map { " \($0)" } ?? ""
            return "Tool\(name):\n\(message.content)"
        }
    }

    private static func renderResponseConstraint(
        _ constraint: MLXBridgeResponseConstraint?
    ) -> String? {
        guard let constraint else {
            return nil
        }
        let instructions = constraint.instructions ?? "Return only JSON that conforms to this schema."
        return """
        Response constraints:
        \(instructions)
        schema:
        \(constraint.jsonSchema)
        """
    }

    static func renderTools(
        _ tools: [MLXBridgeToolDefinition],
        style: MLXPromptStyle
    ) -> String {
        MLXToolPromptDialect(style: style).toolInstructions(for: tools)
    }

    static func chatMLRole(_ message: MLXBridgeMessage) -> String {
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

    private static func glmRole(_ message: MLXBridgeMessage) -> String {
        if message.role == .tool {
            return "observation"
        }
        return chatMLRole(message)
    }

    private static func cohereRole(_ message: MLXBridgeMessage) -> String {
        switch message.role {
        case .system:
            return "SYSTEM"

        case .assistant:
            return "CHATBOT"

        case .user, .tool:
            return "USER"
        }
    }

    private static func cohereTurn(role: String, content: String) -> String {
        "<|START_OF_TURN_TOKEN|><|\(role)_TOKEN|>\(content)<|END_OF_TURN_TOKEN|>"
    }

    private static func gemmaTurn(role: String, content: String) -> String {
        "<start_of_turn>\(role)\n\(content.trimmingCharacters(in: .whitespacesAndNewlines))<end_of_turn>"
    }

    static func toolContent(_ message: MLXBridgeMessage) -> String {
        if let name = message.name, !name.isEmpty {
            return "Tool \(name):\n\(message.content)"
        }
        return message.content
    }

    private static func joinedNonEmpty(_ values: [String]) -> String {
        values
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
    }
}
