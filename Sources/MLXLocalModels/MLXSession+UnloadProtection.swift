extension MLXSession {
    internal func beginGeneration() {
        activeGenerationCount += 1
        MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .generationStarted))
    }

    internal func finishGeneration() {
        guard activeGenerationCount > 0 else {
            logger.fault("Generation lifecycle imbalance: finish called with no active generation")
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(
                stage: .generationFinished,
                message: "finish called with no active generation"
            ))
            return
        }

        activeGenerationCount -= 1
        MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .generationFinished))
        schedulePendingUnloadIfNeeded()
    }

    /// Unload a model from memory.
    internal func unload() async {
        guard !runtimePreferences.isPinned else {
            pendingUnloadAfterGeneration = false
            logger.info("Unload skipped because the model is pinned in runtime preferences")
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(
                stage: .unloadSkipped,
                message: "model is pinned"
            ))
            return
        }

        await pauseAdmissionForUnload()

        if isGenerating {
            pendingUnloadAfterGeneration = true
            logger.info(
                """
                Unload requested during \(self.activeGenerationCount) active generation(s); \
                deferring until all streams finish
                """
            )
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadDeferred))
            return
        }

        if modelContainer != nil {
            logger.info("Unloading model from memory")
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadStarted))
            try? await persistPromptCacheIfNeeded()
            await closeContinuousBatchEngine()
            modelContainer = nil
            memoryProfile = nil
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadFinished))
        } else {
            logger.debug("Unload called but no model was loaded")
            MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadSkipped))
        }

        await resumeAdmissionAfterUnload()
        pendingUnloadAfterGeneration = false
    }

    internal func schedulePendingUnloadIfNeeded() {
        guard pendingUnloadAfterGeneration, !isGenerating else {
            return
        }
        Task {
            await self.unload()
        }
    }

    private func pauseAdmissionForUnload() async {
        await generationAdmission.setAdmissionPaused(true)
        MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadAdmissionPaused))
    }

    private func resumeAdmissionAfterUnload() async {
        await generationAdmission.setAdmissionPaused(false)
        MLXGenerationDiagnostics.recordSessionLifecycle(lifecycleSnapshot(stage: .unloadAdmissionResumed))
    }

    private func lifecycleSnapshot(
        stage: MLXSessionLifecycleSnapshot.Stage,
        message: String? = nil
    ) -> MLXSessionLifecycleSnapshot {
        MLXSessionLifecycleSnapshot(
            stage: stage,
            activeGenerationCount: activeGenerationCount,
            pendingUnloadAfterGeneration: pendingUnloadAfterGeneration,
            hasModelContainer: modelContainer != nil,
            message: message
        )
    }
}
