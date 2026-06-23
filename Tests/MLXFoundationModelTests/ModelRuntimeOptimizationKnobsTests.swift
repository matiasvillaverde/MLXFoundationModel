import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime optimization knobs")
struct ModelRuntimeOptimizationKnobsTests {
    @Test("ExternalDraft runtime knobs round-trip and validate")
    func externalDraftRuntimeKnobsRoundTripAndValidate() throws {
        let optimization = MLXRuntimeOptimizationConfiguration.externalDraft(
            draftModelID: "Qwen3-0.6B-4bit",
            maxContextTokens: 8_192
        )

        let decoded = try Self.roundTrip(optimization)

        #expect(decoded.mode == .externalDraft)
        #expect(decoded.draftModelID == "Qwen3-0.6B-4bit")
        #expect(decoded.maxContextTokens == 8_192)
        try ModelRuntimePreferences(optimization: decoded).validate()
    }

    @Test("SpecPrefill runtime knobs round-trip and validate for dense fallback")
    func specPrefillRuntimeKnobsRoundTripAndValidateForDenseFallback() throws {
        let optimization = MLXRuntimeOptimizationConfiguration.specPrefill(
            draftModelID: "qwen3.5-specprefill-draft",
            maxContextTokens: 12_288,
            keepRate: 0.25,
            thresholdTokens: 4_096
        )

        let decoded = try Self.roundTrip(optimization)

        #expect(decoded.mode == .specPrefill)
        #expect(decoded.draftModelID == "qwen3.5-specprefill-draft")
        #expect(decoded.maxContextTokens == 12_288)
        #expect(decoded.specPrefill?.keepRate == 0.25)
        #expect(decoded.specPrefill?.thresholdTokens == 4_096)
        try ModelRuntimePreferences(optimization: decoded).validate()
    }

    @Test("SpecPrefill runtime knobs normalize user supplied bounds")
    func specPrefillRuntimeKnobsNormalizeUserSuppliedBounds() {
        let low = MLXSpecPrefillRuntimeConfiguration(keepRate: 0.01, thresholdTokens: -10)
        let high = MLXSpecPrefillRuntimeConfiguration(keepRate: 0.9, thresholdTokens: 0)
        let invalid = MLXSpecPrefillRuntimeConfiguration(keepRate: .nan, thresholdTokens: nil)

        #expect(low.keepRate == 0.1)
        #expect(low.thresholdTokens == 1)
        #expect(high.keepRate == 0.5)
        #expect(high.thresholdTokens == 1)
        #expect(invalid.keepRate == nil)
    }

    @Test("DFlash runtime knobs round-trip and fail closed")
    func dFlashRuntimeKnobsRoundTripAndFailClosed() throws {
        let configuration = MLXDFlashRuntimeConfiguration(
            draftWindowSize: 2_048,
            draftSinkSize: 128,
            verifyMode: .adaptive,
            useMemoryCache: true,
            memoryCacheMaxEntries: 6,
            memoryCacheMaxBytes: 1_024,
            useSSDCache: true,
            ssdCacheMaxBytes: 4_096
        )
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft",
            maxContextTokens: 65_536,
            configuration: configuration
        )

        let decoded = try Self.roundTrip(optimization)

        #expect(decoded.mode == .dFlash)
        #expect(decoded.draftModelID == "qwen3.5-dflash-draft")
        #expect(decoded.maxContextTokens == 65_536)
        #expect(decoded.dFlash?.draftWindowSize == 2_048)
        #expect(decoded.dFlash?.draftSinkSize == 128)
        #expect(decoded.dFlash?.verifyMode == .adaptive)
        #expect(decoded.dFlash?.useSSDCache == true)
        #expect(decoded.dFlash?.ssdCacheMaxBytes == 4_096)
        try Self.expectUnsupportedRuntimePath(decoded)
    }

    @Test("DFlash runtime knobs normalize cache and window bounds")
    func dFlashRuntimeKnobsNormalizeCacheAndWindowBounds() {
        let configuration = MLXDFlashRuntimeConfiguration(
            draftWindowSize: 0,
            draftSinkSize: -1,
            memoryCacheMaxEntries: 0,
            memoryCacheMaxBytes: 0,
            ssdCacheMaxBytes: 0
        )

        #expect(configuration.draftWindowSize == 1)
        #expect(configuration.draftSinkSize == 0)
        #expect(configuration.memoryCacheMaxEntries == 1)
        #expect(configuration.memoryCacheMaxBytes == 1)
        #expect(configuration.ssdCacheMaxBytes == 1)
    }

    @Test("SpecPrefill chunk selector preserves prompt order for top chunks")
    func specPrefillChunkSelectorPreservesPromptOrderForTopChunks() {
        let indices = MLXSpecPrefillChunkSelector.selectedTokenIndices(
            importance: [0, 0, 10, 10, 1, 1, 5, 5],
            keepRate: 0.5,
            chunkSize: 2
        )

        #expect(indices == [2, 3, 6, 7])
    }

    @Test("SpecPrefill chunk selector keeps at least one chunk")
    func specPrefillChunkSelectorKeepsAtLeastOneChunk() {
        let indices = MLXSpecPrefillChunkSelector.selectedTokenIndices(
            importance: [0, 1, 7, 7, 2, 2],
            keepRate: 0,
            chunkSize: 2
        )

        #expect(indices == [2, 3])
    }

    @Test("SpecPrefill chunk selector returns all tokens for full keep rate")
    func specPrefillChunkSelectorReturnsAllTokensForFullKeepRate() {
        let indices = MLXSpecPrefillChunkSelector.selectedTokenIndices(
            importance: [1, 2, 3],
            keepRate: 1,
            chunkSize: 2
        )

        #expect(indices == [0, 1, 2])
    }

    @Test("SpecPrefill chunk selector prefers earlier chunks on ties")
    func specPrefillChunkSelectorPrefersEarlierChunksOnTies() {
        let indices = MLXSpecPrefillChunkSelector.selectedTokenIndices(
            importance: [1, 1, 1, 1, 1, 1],
            keepRate: 1.0 / 3.0,
            chunkSize: 2
        )

        #expect(indices == [0, 1])
    }

    private static func roundTrip(
        _ optimization: MLXRuntimeOptimizationConfiguration
    ) throws -> MLXRuntimeOptimizationConfiguration {
        let data = try JSONEncoder().encode(optimization)
        return try JSONDecoder().decode(MLXRuntimeOptimizationConfiguration.self, from: data)
    }

    private static func expectUnsupportedRuntimePath(
        _ optimization: MLXRuntimeOptimizationConfiguration
    ) throws {
        do {
            try ModelRuntimePreferences(optimization: optimization).validate()
            Issue.record("Expected \(optimization.mode.rawValue) to fail closed")
        } catch LLMError.invalidConfiguration(let message) {
            #expect(message.contains("not implemented"))
            #expect(message.contains(optimization.mode.rawValue))
        }
    }
}
