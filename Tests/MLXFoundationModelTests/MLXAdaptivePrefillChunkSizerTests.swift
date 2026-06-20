@testable import MLXLocalModels
import Testing

@Suite("MLX adaptive prefill chunk sizer")
struct MLXAdaptivePrefillChunkSizerTests {
    private static let gib = Int64(1_073_741_824)

    @Test("keeps requested chunk size when predicted peak fits")
    func keepsRequestedChunkSizeWhenPredictedPeakFits() {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: Self.customGuard(limitGiB: 40),
            profile: Self.smallProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            currentMemoryBytes: 10 * Self.gib,
            physicalMemoryBytes: 64 * Self.gib
        )

        #expect(decision.selectedChunkSize == 2_048)
        #expect(decision.snapshot.stage == .unchanged)
        #expect(decision.snapshot.predictedTransientBytes != nil)
    }

    @Test("shrinks a risky chunk from a low memory baseline")
    func shrinksRiskyChunkFromLowBaseline() throws {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: Self.customGuard(limitGiB: 40),
            profile: Self.largeMoEProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            currentMemoryBytes: 20 * Self.gib,
            physicalMemoryBytes: 64 * Self.gib
        )
        let target = try #require(decision.snapshot.targetBytes)
        let current = try #require(decision.snapshot.currentMemoryBytes)
        let predicted = try #require(decision.snapshot.predictedTransientBytes)
        let predictedPerToken = Double(predicted) / 2_048.0
        let selectedTransient = Int64(
            (predictedPerToken * Double(decision.selectedChunkSize)).rounded(.up)
        )
        let selectedPeak = current + selectedTransient

        #expect(decision.snapshot.stage == .adjusted)
        #expect(decision.selectedChunkSize < 2_048)
        #expect(decision.selectedChunkSize >= 32)
        #expect(selectedPeak <= target + Int64(predictedPerToken.rounded(.up)))
    }

    @Test("uses the minimum chunk floor when little headroom remains")
    func usesMinimumChunkFloorWhenHeadroomIsSmall() {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: Self.customGuard(limitGiB: 40),
            profile: Self.largeMoEProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 64,
            currentMemoryBytes: 39 * Self.gib,
            physicalMemoryBytes: 64 * Self.gib
        )

        #expect(decision.snapshot.stage == .adjusted)
        #expect(decision.selectedChunkSize == 64)
    }

    @Test("leaves chunk unchanged when memory guard is disabled")
    func leavesChunkUnchangedWhenMemoryGuardIsDisabled() {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: .off,
            profile: Self.largeMoEProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            currentMemoryBytes: 39 * Self.gib,
            physicalMemoryBytes: 64 * Self.gib
        )

        #expect(decision.snapshot.stage == .disabled)
        #expect(decision.selectedChunkSize == 2_048)
    }

    @Test("leaves chunk unchanged when no usable memory limit exists")
    func leavesChunkUnchangedWhenNoLimitExists() {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: Self.customGuard(limitGiB: 40),
            profile: Self.largeMoEProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            currentMemoryBytes: 39 * Self.gib,
            physicalMemoryBytes: 0
        )

        #expect(decision.snapshot.stage == .limitUnavailable)
        #expect(decision.selectedChunkSize == 2_048)
    }

    @Test("uses measured transient prediction when tracker has samples")
    func usesMeasuredTransientPredictionWhenTrackerHasSamples() throws {
        let measuredPrediction = 20 * Self.gib
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: Self.customGuard(limitGiB: 40),
            profile: Self.smallProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 4_096,
            processedTokenCount: 512,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            currentMemoryBytes: 20 * Self.gib,
            physicalMemoryBytes: 64 * Self.gib,
            measuredPredictedTransientBytes: measuredPrediction,
            observedBytesPerToken: 10_240,
            observedSampleCount: 3
        )

        let predicted = try #require(decision.snapshot.predictedTransientBytes)
        #expect(decision.snapshot.stage == .adjusted)
        #expect(decision.selectedChunkSize < 2_048)
        #expect(predicted == measuredPrediction)
        #expect(decision.snapshot.cachedTokenCount == 4_096)
        #expect(decision.snapshot.processedTokenCount == 512)
        #expect(decision.snapshot.observedSampleCount == 3)
    }

    @Test("tracker updates EWMA and ignores negative deltas")
    func trackerUpdatesEWMAAndIgnoresNegativeDeltas() throws {
        let tracker = MLXAdaptivePrefillTransientTracker()

        tracker.record(tokenCount: 100, transientBytes: 1_000)
        tracker.record(tokenCount: 100, transientBytes: -1_000)
        tracker.record(tokenCount: 100, transientBytes: 2_000)

        let observation = try #require(tracker.observation)
        let prediction = try #require(tracker.prediction(tokenCount: 50, safetyFactor: 1.0))

        #expect(observation.sampleCount == 2)
        #expect(abs(observation.bytesPerToken - 13) < 0.001)
        #expect(prediction == 650)
    }

    @Test("controller carries processed cache state into decisions")
    func controllerCarriesProcessedCacheStateIntoDecisions() {
        let tracker = MLXAdaptivePrefillTransientTracker()
        let controller = MLXAdaptivePrefillChunkController(
            configuration: Self.customGuard(limitGiB: 1_000),
            profile: Self.smallProfile,
            promptTokenCount: 4_096,
            cachedTokenCount: 512,
            requestedChunkSize: 256,
            minimumChunkSize: 32,
            memorySnapshot: MLXRuntimeMemorySnapshot(
                currentMemoryBytes: 1_000,
                cacheMemoryBytes: 0,
                physicalMemoryBytes: 2_000 * Self.gib,
                metalLimitBytes: nil
            ),
            tracker: tracker
        )

        controller.recordChunk(
            tokenCount: 128,
            memoryBeforeBytes: 1_000,
            memoryAfterBytes: 2_280
        )
        let decision = controller.decision(remainingTokenCount: 2_048)

        #expect(decision.snapshot.cachedTokenCount == 640)
        #expect(decision.snapshot.processedTokenCount == 128)
        #expect(decision.snapshot.currentMemoryBytes == 2_280)
        #expect(decision.snapshot.observedSampleCount == 1)
        #expect(decision.snapshot.observedBytesPerToken == 10)
    }

    @Test("host available memory tightens adaptive prefill target")
    func hostAvailableMemoryTightensAdaptivePrefillTarget() throws {
        let decision = MLXAdaptivePrefillChunkSizer.decision(
            configuration: MLXMemoryGuardConfiguration(tier: .balanced, hardLimitFraction: 1),
            profile: Self.largeMoEProfile,
            promptTokenCount: 16_384,
            cachedTokenCount: 0,
            requestedChunkSize: 2_048,
            minimumChunkSize: 32,
            memorySnapshot: MLXRuntimeMemorySnapshot(
                currentMemoryBytes: 20 * Self.gib,
                cacheMemoryBytes: 0,
                physicalMemoryBytes: 64 * Self.gib,
                metalLimitBytes: nil,
                availableMemoryBytes: Self.gib
            )
        )
        let limit = try #require(decision.snapshot.limitBytes)
        let target = try #require(decision.snapshot.targetBytes)

        #expect(decision.snapshot.stage == .adjusted)
        #expect(decision.selectedChunkSize < 2_048)
        #expect(limit == 21 * Self.gib)
        #expect(target == Int64(Double(21 * Self.gib) * 0.9))
    }

    private static var smallProfile: MLXModelMemoryProfile {
        MLXModelMemoryProfile(
            numLayers: 24,
            numKVHeads: 4,
            numAttentionHeads: 16,
            headDimension: 128,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
    }

    private static var largeMoEProfile: MLXModelMemoryProfile {
        MLXModelMemoryProfile(
            numLayers: 400,
            numKVHeads: 32,
            numAttentionHeads: 64,
            headDimension: 512,
            dtypeSize: 2,
            scoreDTypeSize: 2,
            kvBytesPerTokenOverride: nil
        )
    }

    private static func customGuard(limitGiB: Int64) -> MLXMemoryGuardConfiguration {
        MLXMemoryGuardConfiguration(
            tier: .custom,
            customLimitBytes: limitGiB * Self.gib,
            hardLimitFraction: 1
        )
    }
}
