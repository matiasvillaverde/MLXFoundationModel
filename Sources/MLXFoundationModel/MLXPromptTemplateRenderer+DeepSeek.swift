import Foundation

extension MLXPromptTemplateRenderer {
    static func renderDeepSeekDSML(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        let messages = expandedMessages(for: request, style: style)
        let body = deepSeekBody(messages, request: request, style: style)
        return """
        \(deepSeekBOSToken)\
        \(body)\
        \(deepSeekAssistantContinuation(for: messages))\
        \(reasoningOpenSuffix(for: request, style: style))
        """
    }

    private static let deepSeekBOSToken = "<｜begin▁of▁sentence｜>"
    private static let deepSeekEOSToken = "<｜end▁of▁sentence｜>"

    private static func deepSeekBody(
        _ messages: [MLXBridgeMessage],
        request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> String {
        var rendered = ""
        var index = messages.startIndex
        while index < messages.endIndex {
            if messages[index].role == .tool {
                let range = consecutiveToolRange(in: messages, from: index)
                rendered += deepSeekToolResults(
                    messages[range],
                    reasoningEnabled: request.effectiveReasoningOptions.isEnabled
                )
                index = range.upperBound
                continue
            }
            rendered += deepSeekMessage(messages[index], style: style)
            index = messages.index(after: index)
        }
        return rendered
    }

    private static func deepSeekMessage(
        _ message: MLXBridgeMessage,
        style: MLXPromptStyle
    ) -> String {
        switch message.role {
        case .assistant:
            return "\(renderedContent(for: message, style: style))\(deepSeekEOSToken)"

        case .system:
            return message.content

        case .tool:
            return deepSeekToolResult(message)

        case .user:
            return "<｜User｜>\(message.content)<｜Assistant｜>"
        }
    }

    private static func deepSeekToolResult(_ message: MLXBridgeMessage) -> String {
        deepSeekToolResults(ArraySlice([message]), reasoningEnabled: false)
    }

    private static func deepSeekToolResults(
        _ messages: ArraySlice<MLXBridgeMessage>,
        reasoningEnabled: Bool
    ) -> String {
        let results = messages.map { message in
            "<result>\(message.content)</result>"
        }
        .joined(separator: "\n")
        let thinkingMarker = reasoningEnabled ? "<think>\n" : "</think>"
        return """


        <function_results>
        \(results)
        </function_results>

        \(thinkingMarker)
        """
    }

    private static func consecutiveToolRange(
        in messages: [MLXBridgeMessage],
        from startIndex: Int
    ) -> Range<Int> {
        var endIndex = startIndex
        while endIndex < messages.endIndex, messages[endIndex].role == .tool {
            endIndex = messages.index(after: endIndex)
        }
        return startIndex ..< endIndex
    }

    private static func deepSeekAssistantContinuation(
        for messages: [MLXBridgeMessage]
    ) -> String {
        guard messages.last?.role != .user, messages.last?.role != .tool else {
            return ""
        }
        return "<｜Assistant｜>"
    }
}
