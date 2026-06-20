import Foundation
import MLXLocalModels

enum MLXTranslatedStreamEvent: Sendable {
    case reasoningText(String, tokenCount: Int)
    case responseText(String, tokenCount: Int)
    case responseUsage(UsageMetrics)
    case toolCall(MLXExtractedToolCall, tokenCount: Int)
    case toolUsage(UsageMetrics)
}
