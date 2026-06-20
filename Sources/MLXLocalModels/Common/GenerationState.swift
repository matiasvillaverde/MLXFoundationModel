import Foundation

/// State container for text generation
internal final class GenerationState: @unchecked Sendable {
    var allTokens: [Int] = []
    var generatedTokenCount = 0
    var streamedTokenCount = 0
    var firstTokenTime: ContinuousClock.Instant?
    var stopReason: GenerationMetrics.StopReason = .endOfSequence
    var generatedText = ""
    var stopDetector: StopSequenceDetector?
    var detokenizer: NaiveStreamingDetokenizer?

    deinit { /* Clean up */ }
}
