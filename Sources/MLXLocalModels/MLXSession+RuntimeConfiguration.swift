import Foundation

extension MLXSession {
    private struct PersistedPromptCacheEnvelope: Codable {
        let tokens: [Int]
        let signature: PromptCacheSignature
    }

    private static let persistedPromptCacheMetadataKey = "patagonia.prompt_cache.envelope.v1"

    internal func applyRuntimeConfiguration() async throws {
        runtimePreferences = configuration?.runtime ?? .default
        persistentPromptCacheURL = configuration.map(Self.makePersistentPromptCacheURL(for:))
        speculativeDecoding = try await makeSpeculativeDecodingConfigurationIfNeeded()

        guard let modelContainer else {
            return
        }

        if runtimePreferences.promptCachePolicy == .off {
            await modelContainer.clearPromptCache()
            return
        }

        if runtimePreferences.promptCachePolicy == .persistent {
            try await restorePersistentPromptCacheIfNeeded()
        }
    }

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
            Self.persistedPromptCacheMetadataKey: envelopeData.base64EncodedString()
        ]

        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true,
            attributes: nil
        )
        try savePromptCache(url: url, cache: entry.cache, metadata: metadata)
    }

    internal func restorePersistentPromptCacheIfNeeded() async throws {
        guard runtimePreferences.promptCachePolicy == .persistent,
            speculativeDecoding == nil,
            let modelContainer,
            let url = persistentPromptCacheURL,
            FileManager.default.fileExists(atPath: url.path)
        else {
            return
        }

        let (cache, metadata) = try loadPromptCache(url: url)
        guard let encodedEnvelope = metadata?[Self.persistedPromptCacheMetadataKey],
            let envelopeData = Data(base64Encoded: encodedEnvelope)
        else {
            logger.warning("Persistent prompt cache metadata missing for \(url.path)")
            return
        }

        let envelope = try JSONDecoder().decode(PersistedPromptCacheEnvelope.self, from: envelopeData)
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
        guard runtimePreferences.speculativeDecodingMode == .sameModelDraft,
            let modelContainer
        else {
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

    nonisolated private static func makePersistentPromptCacheURL(
        for configuration: ProviderConfiguration
    ) -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first ??
            FileManager.default.temporaryDirectory
        let fingerprintSeed = [
            configuration.modelName,
            configuration.location.standardizedFileURL.path,
            String(configuration.compute.contextSize)
        ].joined(separator: "|")
        let filename = "\(PromptCacheIdentity.stableFingerprint(for: fingerprintSeed)).safetensors"

        return root
            .appendingPathComponent("PatagoniaAppStore", isDirectory: true)
            .appendingPathComponent("PromptCache", isDirectory: true)
            .appendingPathComponent(filename)
    }
}
