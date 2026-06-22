import Foundation

extension MLXPromptTemplateRenderer {
    static func renderedContent(
        for message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        guard message.role == .assistant else {
            return message.content
        }
        let reasoning = renderedReasoningContent(
            message.reasoningContent,
            style: style
        )
        let body = renderedAssistantBody(for: message, style: style)
        return joinedAssistantContent(reasoning: reasoning, body: body, style: style)
    }

    private static func renderedAssistantBody(
        for message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        let calls = MLXToolCallExtractor.extractAll(from: message.content)
        guard !calls.isEmpty else {
            return sanitizedAssistantContent(message.content, style: style)
        }
        return MLXToolPromptDialect(style: style).renderToolCalls(calls)
    }

    private static func sanitizedAssistantContent(
        _ content: String,
        style: MLXPromptStyle
    ) -> String {
        style == .gemma ? MLXGemma4HistorySanitizer.sanitize(content) : content
    }

    private static func renderedReasoningContent(
        _ reasoningContent: String?,
        style: MLXPromptStyle
    ) -> String {
        guard let reasoningContent, !reasoningContent.isEmpty else {
            return ""
        }
        if style == .gemma {
            return "<|channel>thought\n\(reasoningContent)<channel|>"
        }
        return "Reasoning:\n\(reasoningContent)"
    }

    private static func joinedAssistantContent(
        reasoning: String,
        body: String,
        style: MLXPromptStyle
    ) -> String {
        guard !reasoning.isEmpty else {
            return body
        }
        guard !body.isEmpty else {
            return reasoning
        }
        return style == .gemma ? reasoning + body : "\(reasoning)\n\n\(body)"
    }
}
