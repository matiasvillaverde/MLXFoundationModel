import Foundation

extension MLXSession {
    private struct PersistedPromptCacheEnvelope: Codable {
        let tokens: [Int]
        let signature: PromptCacheSignature
    }

    internal func applyRuntimeConfiguration() async throws {
        try await configureRuntimePreferences()
        speculativeDecoding = try await makeSpeculativeDecodingConfigurationIfNeeded()
        if let configuration {
            memoryProfile = try? MLXModelMemoryProfile.load(modelDirectory: configuration.location)
        } else {
            memoryProfile = nil
        }

        guard let modelContainer else {
            return
        }

        await refineMemoryProfileWithLiveCacheLayout(modelContainer)
        await configureContinuousBatchEngine(for: modelContainer)

        if runtimePreferences.promptCachePolicy == .off {
            await modelContainer.clearPromptCache()
            return
        }

        if runtimePreferences.promptCachePolicy == .persistent {
            try MLXPersistentPromptCacheBudgetEnforcer.enforceAll(
                limitBytes: runtimePreferences.persistentPromptCacheTotalByteLimit,
                protectedSnapshotURL: persistentPromptCacheURL
            )
            try await restorePersistentPromptCacheIfNeeded()
        }
    }

    internal func preflightRuntimeForModelLoad() async throws {
        try await configureRuntimePreferences()
        guard let configuration else {
            return
        }
        try MLXRuntimeMemoryGuard.preflightModelLoad(
            configuration: runtimePreferences.memoryGuard,
            modelDirectory: configuration.location
        )
    }

    private func configureRuntimePreferences() async throws {
        runtimePreferences = configuration?.runtime ?? .default
        let executionPlan = try MLXGenerationExecutionPlanner.plan(
            preferences: runtimePreferences,
            capabilities: runtimeCapabilities
        )
        generationExecutionPlan = executionPlan
        MLXGenerationDiagnostics.recordExecutionPlan(executionPlan)
        if executionPlan.downgradedToScalar {
            logger.warning(
                "Continuous batching is not active in the scalar MLX engine; using serial admission"
            )
        }
        await generationAdmission.updateConfiguration(executionPlan.effectiveScheduling)
        MLXPersistentPromptCacheBlockStore.configureHotCache(
            limitBytes: runtimePreferences.persistentPromptCacheHotByteLimit
        )
        persistentPromptCacheURL = configuration.map(MLXPersistentPromptCacheStore.url(for:))
    }

    private func refineMemoryProfileWithLiveCacheLayout(_ modelContainer: ModelContainer) async {
        guard let profile = memoryProfile else {
            return
        }
        memoryProfile = await modelContainer.perform { context in
            profile.refinedWithLiveCacheLayout(context.model.newCache(parameters: nil))
        }
    }

    // swiftlint:disable:next function_body_length
    internal func persistPromptCacheIfNeeded() async throws {
        guard runtimePreferences.promptCachePolicy == .persistent,
            speculativeDecoding == nil,
            let modelContainer,
            let url = persistentPromptCacheURL
        else {
            return
        }

        let entries = await modelContainer.promptCacheEntriesSnapshot()
        guard let entry = entries.first,
            entry.draftCache == nil,
            entry.byteCount <= runtimePreferences.promptCacheByteLimit
        else {
            return
        }

        let envelope = PersistedPromptCacheEnvelope(
            tokens: entry.tokens,
            signature: entry.signature
        )
        let envelopeData = try JSONEncoder().encode(envelope)
        let metadata = [
            MLXPersistentPromptCacheStore.metadataKey: envelopeData.base64EncodedString()
        ]

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        try savePromptCache(url: url, cache: entry.cache, metadata: metadata)
        try MLXPersistentPromptCacheBudgetEnforcer.enforceAll(
            limitBytes: runtimePreferences.persistentPromptCacheTotalByteLimit,
            protectedSnapshotURL: url
        )
    }

    // swiftlint:disable:next function_body_length
    internal func restorePersistentPromptCacheIfNeeded() async throws {
        guard runtimePreferences.promptCachePolicy == .persistent,
            speculativeDecoding == nil,
            let modelContainer,
            let url = persistentPromptCacheURL,
            FileManager.default.fileExists(atPath: url.path)
        else {
            return
        }

        let cache: [KVCache]
        let envelope: PersistedPromptCacheEnvelope
        do {
            let payload = try loadPromptCache(url: url)
            guard let encodedEnvelope = payload.1?[MLXPersistentPromptCacheStore.metadataKey],
                let envelopeData = Data(base64Encoded: encodedEnvelope)
            else {
                throw LLMError.invalidConfiguration("Persistent prompt cache metadata missing")
            }
            envelope = try JSONDecoder().decode(PersistedPromptCacheEnvelope.self, from: envelopeData)
            cache = payload.0
        } catch {
            logger.warning(
                "Skipping corrupt persistent prompt cache at \(url.path): \(error.localizedDescription)"
            )
            try? FileManager.default.removeItem(at: url)
            return
        }

        let byteCount = PromptCachePlanner.cacheByteCount(cache)
        guard byteCount <= runtimePreferences.promptCacheByteLimit else {
            logger.info("Skipping oversized persistent prompt cache restore for \(url.path)")
            return
        }

        await modelContainer.replacePromptCacheEntries([
            PromptCacheEntry(
                tokens: envelope.tokens,
                cache: cache,
                signature: envelope.signature,
                byteCount: byteCount
            )
        ])
    }

    internal func makeSpeculativeDecodingConfigurationIfNeeded() async throws
        -> MLXSpeculativeDecodingConfiguration? {
        if runtimePreferences.optimization.mode == .externalDraft
            || runtimePreferences.optimization.mode == .vlmMTP {
            guard let draftModelID = runtimePreferences.optimization.draftModelID,
                !draftModelID.isEmpty else {
                return nil
            }
            let draftContext = try await LLMModelFactory.shared.load(
                configuration: draftModelConfiguration(for: draftModelID)
            )
            return MLXSpeculativeDecodingConfiguration(
                draftContext: draftContext,
                numDraftTokens: max(1, runtimePreferences.speculativeDraftTokens)
            )
        }

        guard runtimePreferences.speculativeDecodingMode == .sameModelDraft,
            let modelContainer else {
            return nil
        }

        let draftContext = await modelContainer.perform { context in
            context
        }
        return MLXSpeculativeDecodingConfiguration(
            draftContext: draftContext,
            numDraftTokens: max(1, runtimePreferences.speculativeDraftTokens)
        )
    }

    private func draftModelConfiguration(for draftModelID: String) -> ModelConfiguration {
        let expandedPath = expandedTildePath(draftModelID)
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: expandedPath, isDirectory: &isDirectory),
            isDirectory.boolValue {
            return ModelConfiguration(directory: URL(fileURLWithPath: expandedPath, isDirectory: true))
        }

        if let sibling = configuration?.location.deletingLastPathComponent()
            .appendingPathComponent(draftModelID, isDirectory: true),
            FileManager.default.fileExists(atPath: sibling.path, isDirectory: &isDirectory),
            isDirectory.boolValue {
            return ModelConfiguration(directory: sibling)
        }

        return ModelConfiguration(id: draftModelID)
    }

    private func expandedTildePath(_ path: String) -> String {
        guard path.hasPrefix("~/") else {
            return path
        }
        return FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(String(path.dropFirst(2)))
            .path
    }
}
