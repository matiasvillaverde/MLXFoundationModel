import Foundation

struct MLXTranslatedStreamEventBuilder {
    private var thinkSplitter: MLXThinkTagStreamSplitter
    private var pendingResponseTokenCount = 0

    init(reasoningStartsOpen: Bool = false) {
        thinkSplitter = MLXThinkTagStreamSplitter(startInReasoning: reasoningStartsOpen)
    }

    mutating func events(
        from action: MLXToolAwareStreamReducer.Action
    ) -> [MLXTranslatedStreamEvent] {
        switch action {
        case let .responseText(text, tokenCount):
            pendingResponseTokenCount += max(tokenCount, 1)
            return textEvents(from: thinkSplitter.consume(text))

        case .responseUsage(let usage):
            return finish() + [.responseUsage(usage)]

        case let .toolCall(call, tokenCount):
            return finish() + [.toolCall(call, tokenCount: tokenCount)]

        case .toolUsage(let usage):
            return finish() + [.toolUsage(usage)]
        }
    }

    mutating func finish() -> [MLXTranslatedStreamEvent] {
        let events = textEvents(from: thinkSplitter.finish())
        pendingResponseTokenCount = 0
        return events
    }

    private mutating func textEvents(
        from segments: [MLXThinkTagStreamSplitter.Segment]
    ) -> [MLXTranslatedStreamEvent] {
        guard !segments.isEmpty else {
            return []
        }

        let tokenBudget = emittedTokenBudget(for: segments)
        pendingResponseTokenCount -= tokenBudget
        return allocatedSegments(segments, tokenBudget: tokenBudget).map { item in
            switch item.segment.kind {
            case .reasoning:
                return .reasoningText(item.segment.text, tokenCount: item.tokenCount)

            case .response:
                return .responseText(item.segment.text, tokenCount: item.tokenCount)
            }
        }
    }

    private func emittedTokenBudget(
        for segments: [MLXThinkTagStreamSplitter.Segment]
    ) -> Int {
        let retainedByteCount = thinkSplitter.retainedUTF8ByteCount
        let emittedByteCount = segments.reduce(0) { $0 + max($1.text.utf8.count, 1) }
        let minimumRetainedTokens = retainedByteCount > 0 ? 1 : 0
        let availableTokens = max(
            pendingResponseTokenCount,
            segments.count + minimumRetainedTokens
        )
        guard retainedByteCount > 0 else {
            return availableTokens
        }

        let totalByteCount = max(emittedByteCount + retainedByteCount, 1)
        let proportionalRetainedTokens = Int(
            (Double(retainedByteCount) / Double(totalByteCount)) * Double(availableTokens)
        )
        let retainedTokens = min(
            max(proportionalRetainedTokens, minimumRetainedTokens),
            max(availableTokens - segments.count, minimumRetainedTokens)
        )
        return availableTokens - retainedTokens
    }

    private func allocatedSegments(
        _ segments: [MLXThinkTagStreamSplitter.Segment],
        tokenBudget: Int
    ) -> [(segment: MLXThinkTagStreamSplitter.Segment, tokenCount: Int)] {
        let totalLength = segments.reduce(0) { $0 + max($1.text.utf8.count, 1) }
        guard totalLength > 0 else {
            return []
        }

        var remainingTokens = max(tokenBudget, segments.count)
        var remainingLength = totalLength
        return segments.enumerated().map { index, segment in
            let length = max(segment.text.utf8.count, 1)
            let tokenCount = allocatedTokenCount(
                length: length,
                remainingLength: remainingLength,
                remainingTokens: remainingTokens,
                remainingSegments: segments.count - index
            )
            remainingTokens -= tokenCount
            remainingLength -= length
            return (segment, tokenCount)
        }
    }

    private func allocatedTokenCount(
        length: Int,
        remainingLength: Int,
        remainingTokens: Int,
        remainingSegments: Int
    ) -> Int {
        guard remainingSegments > 1 else {
            return max(remainingTokens, 1)
        }

        let proportional = Int((Double(length) / Double(remainingLength)) * Double(remainingTokens))
        let maximum = max(remainingTokens - (remainingSegments - 1), 1)
        return min(max(proportional, 1), maximum)
    }
}
