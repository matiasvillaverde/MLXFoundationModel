import Foundation
import MLXLocalModels

struct MLXToolAwareStreamReducer {
    enum Action {
        case responseText(String, tokenCount: Int)
        case responseUsage(UsageMetrics)
        case toolCall(MLXExtractedToolCall, tokenCount: Int)
        case toolUsage(UsageMetrics)
    }

    private var accumulatedText = ""
    private var protocolNormalizer: MLXProtocolStreamNormalizerPipeline
    private var filter: MLXToolCallStreamFilter
    private var latestMetrics: ChunkMetrics?
    private var pendingTokenCount = 0
    private var emittedToolCallKeys = Set<String>()
    private var emittedToolCallCountsByFingerprint: [String: Int] = [:]
    private let tools: [MLXBridgeToolDefinition]
    private let validToolNames: Set<String>

    init(
        tools: [MLXBridgeToolDefinition],
        promptStyle: MLXPromptStyle? = nil
    ) {
        self.tools = tools
        validToolNames = Set(tools.map(\.name))
        protocolNormalizer = MLXProtocolStreamNormalizerPipeline(promptStyle: promptStyle)
        filter = MLXToolCallStreamFilter(toolNames: validToolNames)
    }

    mutating func consume(_ chunk: LLMStreamChunk) -> [Action] {
        latestMetrics = chunk.metrics ?? latestMetrics
        guard !chunk.text.isEmpty else {
            return []
        }

        accumulatedText += chunk.text
        pendingTokenCount += Self.tokenCount(for: chunk)
        let protocolVisibleText = normalizeProtocolText(chunk.text)
        if tools.isEmpty {
            guard !protocolVisibleText.isEmpty else {
                return []
            }
            return [.responseText(protocolVisibleText, tokenCount: consumePendingTokenCount())]
        }

        var actions: [Action] = []
        let visibleText = filter.feed(protocolVisibleText)
        if !visibleText.isEmpty {
            actions.append(.responseText(visibleText, tokenCount: consumePendingTokenCount()))
        }
        actions.append(contentsOf: newToolCallActionsFromCompletedEnvelopes())
        return actions
    }

    mutating func finish() -> [Action] {
        if tools.isEmpty {
            var actions = protocolRemainderActions()
            actions.append(contentsOf: usageAction { .responseUsage($0) })
            return actions
        }

        var actions = filteredRemainderActions()
        var toolActions = newToolCallActionsFromCompletedEnvelopes()
        toolActions.append(contentsOf: remainingToolCallActionsFromAccumulatedText())
        guard !toolActions.isEmpty || !emittedToolCallKeys.isEmpty else {
            actions.append(contentsOf: usageAction { .responseUsage($0) })
            return actions
        }

        actions.append(contentsOf: toolActions)
        actions.append(contentsOf: usageAction { .toolUsage($0) })
        return actions
    }

    private mutating func protocolRemainderActions() -> [Action] {
        let visibleText = finishProtocolText()
        guard !visibleText.isEmpty else {
            return []
        }
        return [.responseText(visibleText, tokenCount: consumePendingTokenCount())]
    }

    private mutating func filteredRemainderActions() -> [Action] {
        let protocolVisibleText = finishProtocolText()
        let visibleText = filter.feed(protocolVisibleText) + filter.finish()
        guard !visibleText.isEmpty else {
            return []
        }
        return [.responseText(visibleText, tokenCount: consumePendingTokenCount())]
    }

    private mutating func normalizeProtocolText(_ text: String) -> String {
        protocolNormalizer.feed(text)
    }

    private mutating func finishProtocolText() -> String {
        protocolNormalizer.finish()
    }

    private func usageAction(
        _ makeAction: (UsageMetrics) -> Action
    ) -> [Action] {
        guard let usage = latestMetrics?.usage else {
            return []
        }
        return [makeAction(usage)]
    }

    private mutating func newToolCallActionsFromCompletedEnvelopes() -> [Action] {
        let calls = filter.takeCompletedSuppressedTexts().flatMap { text in
            MLXToolCallExtractor.extractAll(from: text, tools: tools)
        }
        let newCalls = calls.filter { call in
            guard isKnownToolCall(call) else {
                return false
            }
            let key = nextToolCallKey(for: call)
            return emittedToolCallKeys.insert(key).inserted
        }
        return toolCallActions(for: newCalls)
    }

    private mutating func remainingToolCallActionsFromAccumulatedText() -> [Action] {
        let calls = MLXToolCallExtractor.extractAll(from: accumulatedText, tools: tools)
        var localCounts: [String: Int] = [:]
        let newCalls = calls.filter { call in
            guard isKnownToolCall(call) else {
                return false
            }
            let fingerprint = toolCallFingerprint(call)
            let occurrence = localCounts[fingerprint, default: 0]
            localCounts[fingerprint] = occurrence + 1
            let key = toolCallKey(fingerprint: fingerprint, occurrence: occurrence)
            updateEmittedCount(fingerprint: fingerprint, occurrence: occurrence)
            return emittedToolCallKeys.insert(key).inserted
        }
        return toolCallActions(for: newCalls)
    }

    private func isKnownToolCall(_ call: MLXExtractedToolCall) -> Bool {
        validToolNames.contains(call.name)
    }

    private mutating func nextToolCallKey(for call: MLXExtractedToolCall) -> String {
        let fingerprint = toolCallFingerprint(call)
        let occurrence = emittedToolCallCountsByFingerprint[fingerprint, default: 0]
        emittedToolCallCountsByFingerprint[fingerprint] = occurrence + 1
        return toolCallKey(fingerprint: fingerprint, occurrence: occurrence)
    }

    private mutating func updateEmittedCount(fingerprint: String, occurrence: Int) {
        let nextCount = occurrence + 1
        let currentCount = emittedToolCallCountsByFingerprint[fingerprint, default: 0]
        emittedToolCallCountsByFingerprint[fingerprint] = max(currentCount, nextCount)
    }

    private func toolCallFingerprint(_ call: MLXExtractedToolCall) -> String {
        "\(call.name)\u{0}\(call.argumentsJSON)"
    }

    private func toolCallKey(fingerprint: String, occurrence: Int) -> String {
        "\(occurrence)\u{0}\(fingerprint)"
    }

    private mutating func toolCallActions(for calls: [MLXExtractedToolCall]) -> [Action] {
        guard !calls.isEmpty else {
            return []
        }
        let tokenCount = consumePendingTokenCount()
        return calls.enumerated().map { index, call in
            .toolCall(call, tokenCount: index == 0 ? tokenCount : 1)
        }
    }

    private mutating func consumePendingTokenCount() -> Int {
        defer {
            pendingTokenCount = 0
        }
        return max(pendingTokenCount, 1)
    }

    private static func tokenCount(for chunk: LLMStreamChunk) -> Int {
        if chunk.tokenCount > 0 {
            return chunk.tokenCount
        }
        return chunk.text.isEmpty ? 0 : 1
    }
}
