import Foundation

extension MLXPromptTemplateRenderer {
    static func renderedContent(
        for message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        guard message.role == .assistant else {
            return message.content
        }
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
}
