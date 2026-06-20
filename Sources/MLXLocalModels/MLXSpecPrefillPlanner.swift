protocol MLXSpecPrefillSelectionStrategy: Sendable {
    func selectedTokenIndices(
        importance: [Float],
        keepRate: Double,
        chunkSize: Int
    ) -> [Int]
}

struct MLXChunkedSpecPrefillSelectionStrategy: MLXSpecPrefillSelectionStrategy {
    func selectedTokenIndices(
        importance: [Float],
        keepRate: Double,
        chunkSize: Int
    ) -> [Int] {
        MLXSpecPrefillChunkSelector.selectedTokenIndices(
            importance: importance,
            keepRate: keepRate,
            chunkSize: chunkSize
        )
    }
}

struct MLXSpecPrefillPlan: Sendable, Equatable {
    let promptTokenCount: Int
    let cachedTokenCount: Int
    let protectedPrefixTokenCount: Int
    let retainedTokenIndices: [Int]
    let newPrefillTokenIndices: [Int]
    let keepRate: Double
    let thresholdTokens: Int
    let chunkSize: Int

    var decodePositionOffset: Int {
        promptTokenCount - retainedTokenCount
    }

    var retainedTokenCount: Int {
        retainedTokenIndices.count
    }

    var newPrefillTokenCount: Int {
        newPrefillTokenIndices.count
    }
}

enum MLXSpecPrefillPlanner {
    static let defaultKeepRate = 0.2
    static let defaultThresholdTokens = 8_192
    static let defaultChunkSize = 32

    static func plan(
        promptTokenCount: Int,
        cachedTokenCount: Int,
        protectedPrefixTokenCount: Int,
        importance: [Float],
        configuration: MLXSpecPrefillRuntimeConfiguration?,
        selectionStrategy: any MLXSpecPrefillSelectionStrategy =
            MLXChunkedSpecPrefillSelectionStrategy()
    ) -> MLXSpecPrefillPlan? {
        guard let configuration else {
            recordSkipped(
                stage: .skippedDisabled,
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                protectedPrefixTokenCount: protectedPrefixTokenCount,
                message: "SpecPrefill is not configured"
            )
            return nil
        }

        let thresholdTokens = configuration.thresholdTokens ?? defaultThresholdTokens
        let keepRate = configuration.keepRate ?? defaultKeepRate
        let promptTokenCount = max(0, promptTokenCount)
        let cachedTokenCount = min(max(0, cachedTokenCount), promptTokenCount)
        let protectedPrefixTokenCount = min(
            max(0, protectedPrefixTokenCount),
            promptTokenCount
        )

        guard promptTokenCount >= thresholdTokens else {
            recordSkipped(
                stage: .skippedBelowThreshold,
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                protectedPrefixTokenCount: protectedPrefixTokenCount,
                keepRate: keepRate,
                thresholdTokens: thresholdTokens,
                message: "Prompt is below the SpecPrefill threshold"
            )
            return nil
        }

        guard importance.count == promptTokenCount else {
            recordSkipped(
                stage: .skippedImportanceMismatch,
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                protectedPrefixTokenCount: protectedPrefixTokenCount,
                keepRate: keepRate,
                thresholdTokens: thresholdTokens,
                message: "Importance score count does not match prompt token count"
            )
            return nil
        }

        let selected = selectionStrategy.selectedTokenIndices(
            importance: importance,
            keepRate: keepRate,
            chunkSize: defaultChunkSize
        )
        let retained = retainedTokenIndices(
            selected: selected,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            protectedPrefixTokenCount: protectedPrefixTokenCount
        )
        let newPrefill = retained.filter { $0 >= cachedTokenCount }
        let uncachedTokenCount = promptTokenCount - cachedTokenCount
        guard newPrefill.count < uncachedTokenCount else {
            recordSkipped(
                stage: .skippedNoReduction,
                promptTokenCount: promptTokenCount,
                cachedTokenCount: cachedTokenCount,
                protectedPrefixTokenCount: protectedPrefixTokenCount,
                retainedTokenCount: retained.count,
                newPrefillTokenCount: newPrefill.count,
                keepRate: keepRate,
                thresholdTokens: thresholdTokens,
                chunkSize: defaultChunkSize,
                message: "SpecPrefill retained every uncached token"
            )
            return nil
        }

        let plan = MLXSpecPrefillPlan(
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            protectedPrefixTokenCount: protectedPrefixTokenCount,
            retainedTokenIndices: retained,
            newPrefillTokenIndices: newPrefill,
            keepRate: keepRate,
            thresholdTokens: thresholdTokens,
            chunkSize: defaultChunkSize
        )
        MLXGenerationDiagnostics.recordSpecPrefillPlan(.init(
            stage: .planned,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            protectedPrefixTokenCount: protectedPrefixTokenCount,
            retainedTokenCount: plan.retainedTokenCount,
            newPrefillTokenCount: plan.newPrefillTokenCount,
            decodePositionOffset: plan.decodePositionOffset,
            keepRate: keepRate,
            thresholdTokens: thresholdTokens,
            chunkSize: defaultChunkSize,
            message: nil
        ))
        return plan
    }

    static func recordRuntimeUnavailable(
        promptTokenCount: Int,
        cachedTokenCount: Int,
        protectedPrefixTokenCount: Int,
        configuration: MLXSpecPrefillRuntimeConfiguration?
    ) {
        let thresholdTokens = configuration?.thresholdTokens ?? defaultThresholdTokens
        let keepRate = configuration?.keepRate ?? defaultKeepRate
        let normalizedPromptTokenCount = max(0, promptTokenCount)
        let normalizedCachedTokenCount = min(
            max(0, cachedTokenCount),
            normalizedPromptTokenCount
        )
        let normalizedProtectedPrefixTokenCount = min(
            max(0, protectedPrefixTokenCount),
            normalizedPromptTokenCount
        )

        recordSkipped(
            stage: .skippedRuntimeUnavailable,
            promptTokenCount: normalizedPromptTokenCount,
            cachedTokenCount: normalizedCachedTokenCount,
            protectedPrefixTokenCount: normalizedProtectedPrefixTokenCount,
            keepRate: keepRate,
            thresholdTokens: thresholdTokens,
            chunkSize: defaultChunkSize,
            message: "SpecPrefill sparse RoPE runtime is not available; using dense prefill"
        )
    }

    private static func retainedTokenIndices(
        selected: [Int],
        promptTokenCount: Int,
        cachedTokenCount: Int,
        protectedPrefixTokenCount: Int
    ) -> [Int] {
        var retained = Set<Int>()
        retained.reserveCapacity(selected.count + max(cachedTokenCount, protectedPrefixTokenCount))
        for index in selected where index >= 0 && index < promptTokenCount {
            retained.insert(index)
        }
        for index in 0..<max(cachedTokenCount, protectedPrefixTokenCount) {
            retained.insert(index)
        }
        return retained.sorted()
    }

    private static func recordSkipped(
        stage: MLXSpecPrefillPlanSnapshot.Stage,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        protectedPrefixTokenCount: Int,
        retainedTokenCount: Int = 0,
        newPrefillTokenCount: Int = 0,
        decodePositionOffset: Int = 0,
        keepRate: Double? = nil,
        thresholdTokens: Int? = nil,
        chunkSize: Int? = nil,
        message: String
    ) {
        MLXGenerationDiagnostics.recordSpecPrefillPlan(.init(
            stage: stage,
            promptTokenCount: max(0, promptTokenCount),
            cachedTokenCount: max(0, cachedTokenCount),
            protectedPrefixTokenCount: max(0, protectedPrefixTokenCount),
            retainedTokenCount: retainedTokenCount,
            newPrefillTokenCount: newPrefillTokenCount,
            decodePositionOffset: decodePositionOffset,
            keepRate: keepRate,
            thresholdTokens: thresholdTokens,
            chunkSize: chunkSize,
            message: message
        ))
    }
}
