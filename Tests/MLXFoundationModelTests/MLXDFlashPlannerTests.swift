@testable import MLXLocalModels
import Testing

@Suite("MLX DFlash planner")
struct MLXDFlashPlannerTests {
    @Test("plans DFlash runtime settings")
    func plansDFlashRuntimeSettings() async throws {
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft",
            maxContextTokens: 8_192,
            configuration: Self.configuredDFlashRuntime
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 4_096, optimization: optimization)
        }

        let plan = try #require(recorded.result)
        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        Self.expectConfiguredDFlashPlan(plan)
        Self.expectConfiguredDFlashSnapshot(snapshot)
    }

    @Test("defaults DFlash verifier to adaptive")
    func defaultsDFlashVerifierToAdaptive() async throws {
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft"
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 32, optimization: optimization)
        }

        let plan = try #require(recorded.result)
        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(plan.verifyMode == .adaptive)
        #expect(snapshot.verifyMode == .adaptive)
        #expect(!plan.usesSSDCache)
    }

    @Test("skips DFlash when prompt reaches max context threshold")
    func skipsDFlashWhenPromptReachesMaxContextThreshold() async throws {
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft",
            maxContextTokens: 128
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 128, optimization: optimization)
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedContextTooLong)
        #expect(snapshot.maxContextTokens == 128)
        #expect(snapshot.message?.contains("threshold") == true)
    }

    @Test("skips DFlash when disabled")
    func skipsDFlashWhenDisabled() async throws {
        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 128, optimization: .off)
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedDisabled)
    }

    @Test("skips DFlash without draft model")
    func skipsDFlashWithoutDraftModel() async throws {
        let optimization = MLXRuntimeOptimizationConfiguration(mode: .dFlash)

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 128, optimization: optimization)
        }

        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(recorded.result == nil)
        #expect(snapshot.stage == .skippedMissingDraft)
        #expect(snapshot.message?.contains("draft") == true)
    }

    @Test("does not enable SSD cache when memory cache is disabled")
    func doesNotEnableSSDCacheWhenMemoryCacheIsDisabled() async throws {
        let optimization = MLXRuntimeOptimizationConfiguration.dFlash(
            draftModelID: "qwen3.5-dflash-draft",
            configuration: .init(useMemoryCache: false, useSSDCache: true)
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXDFlashPlanner.plan(promptTokenCount: 128, optimization: optimization)
        }

        let plan = try #require(recorded.result)
        let snapshot = try #require(Self.snapshots(from: recorded.events).last)

        #expect(!plan.usesMemoryCache)
        #expect(!plan.usesSSDCache)
        #expect(!snapshot.usesMemoryCache)
        #expect(!snapshot.usesSSDCache)
    }

    private static func snapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXDFlashPlanSnapshot] {
        events.compactMap { event in
            guard case .dFlashPlan(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static var configuredDFlashRuntime: MLXDFlashRuntimeConfiguration {
        MLXDFlashRuntimeConfiguration(
            draftWindowSize: 2_048,
            draftSinkSize: 128,
            verifyMode: .ddTree,
            useMemoryCache: true,
            memoryCacheMaxEntries: 6,
            memoryCacheMaxBytes: 1_024,
            useSSDCache: true,
            ssdCacheMaxBytes: 4_096
        )
    }

    private static func expectConfiguredDFlashPlan(_ plan: MLXDFlashPlan) {
        #expect(plan.draftModelID == "qwen3.5-dflash-draft")
        #expect(plan.verifyMode == .ddTree)
        #expect(plan.draftWindowSize == 2_048)
        #expect(plan.draftSinkSize == 128)
        #expect(plan.usesMemoryCache)
        #expect(plan.usesSSDCache)
        #expect(plan.memoryCacheMaxEntries == 6)
        #expect(plan.memoryCacheMaxBytes == 1_024)
        #expect(plan.ssdCacheMaxBytes == 4_096)
    }

    private static func expectConfiguredDFlashSnapshot(_ snapshot: MLXDFlashPlanSnapshot) {
        #expect(snapshot.stage == .planned)
        #expect(snapshot.verifyMode == .ddTree)
        #expect(snapshot.usesSSDCache)
    }
}
