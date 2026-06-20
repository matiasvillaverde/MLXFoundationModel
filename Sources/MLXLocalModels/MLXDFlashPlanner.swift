struct MLXDFlashPlan: Sendable, Equatable {
    let promptTokenCount: Int
    let draftModelID: String
    let maxContextTokens: Int?
    let draftWindowSize: Int?
    let draftSinkSize: Int?
    let verifyMode: MLXDFlashVerifyMode
    let usesMemoryCache: Bool
    let memoryCacheMaxEntries: Int
    let memoryCacheMaxBytes: Int
    let usesSSDCache: Bool
    let ssdCacheMaxBytes: Int
}

enum MLXDFlashPlanner {
    static let defaultVerifyMode = MLXDFlashVerifyMode.adaptive

    static func plan(
        promptTokenCount: Int,
        optimization: MLXRuntimeOptimizationConfiguration
    ) -> MLXDFlashPlan? {
        guard optimization.mode == .dFlash else {
            recordSkipped(
                stage: .skippedDisabled,
                promptTokenCount: promptTokenCount,
                optimization: optimization,
                message: "DFlash is not configured"
            )
            return nil
        }

        guard let draftModelID = optimization.draftModelID,
              !draftModelID.isEmpty else {
            recordSkipped(
                stage: .skippedMissingDraft,
                promptTokenCount: promptTokenCount,
                optimization: optimization,
                message: "DFlash requires a draft model identifier"
            )
            return nil
        }

        let promptTokenCount = max(0, promptTokenCount)
        if let maxContextTokens = optimization.maxContextTokens,
           promptTokenCount >= maxContextTokens {
            recordSkipped(
                stage: .skippedContextTooLong,
                promptTokenCount: promptTokenCount,
                optimization: optimization,
                message: "Prompt exceeds DFlash context threshold"
            )
            return nil
        }

        let configuration = optimization.dFlash ?? .balanced
        let plan = MLXDFlashPlan(
            promptTokenCount: promptTokenCount,
            draftModelID: draftModelID,
            maxContextTokens: optimization.maxContextTokens,
            draftWindowSize: configuration.draftWindowSize,
            draftSinkSize: configuration.draftSinkSize,
            verifyMode: configuration.verifyMode ?? defaultVerifyMode,
            usesMemoryCache: configuration.useMemoryCache,
            memoryCacheMaxEntries: configuration.memoryCacheMaxEntries,
            memoryCacheMaxBytes: configuration.memoryCacheMaxBytes,
            usesSSDCache: configuration.useMemoryCache && configuration.useSSDCache,
            ssdCacheMaxBytes: configuration.ssdCacheMaxBytes
        )
        MLXGenerationDiagnostics.recordDFlashPlan(snapshot(
            stage: .planned,
            promptTokenCount: promptTokenCount,
            optimization: optimization,
            configuration: configuration,
            draftModelID: draftModelID,
            message: nil
        ))
        return plan
    }

    private static func recordSkipped(
        stage: MLXDFlashPlanSnapshot.Stage,
        promptTokenCount: Int,
        optimization: MLXRuntimeOptimizationConfiguration,
        message: String
    ) {
        MLXGenerationDiagnostics.recordDFlashPlan(snapshot(
            stage: stage,
            promptTokenCount: max(0, promptTokenCount),
            optimization: optimization,
            configuration: optimization.dFlash,
            draftModelID: optimization.draftModelID,
            message: message
        ))
    }

    private static func snapshot(
        stage: MLXDFlashPlanSnapshot.Stage,
        promptTokenCount: Int,
        optimization: MLXRuntimeOptimizationConfiguration,
        configuration: MLXDFlashRuntimeConfiguration?,
        draftModelID: String?,
        message: String?
    ) -> MLXDFlashPlanSnapshot {
        MLXDFlashPlanSnapshot(
            stage: stage,
            promptTokenCount: max(0, promptTokenCount),
            draftModelID: draftModelID,
            maxContextTokens: optimization.maxContextTokens,
            draftWindowSize: configuration?.draftWindowSize,
            draftSinkSize: configuration?.draftSinkSize,
            verifyMode: configuration?.verifyMode ?? (stage == .planned ? defaultVerifyMode : nil),
            usesMemoryCache: configuration?.useMemoryCache ?? false,
            usesSSDCache: (configuration?.useMemoryCache ?? false) && (configuration?.useSSDCache ?? false),
            message: message
        )
    }
}
