import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("MLX reasoning budget processor")
struct MLXReasoningBudgetProcessorTests {
    @Test("state forces a multi-token end marker after the budget")
    func stateForcesMultiTokenEndMarkerAfterBudget() {
        var state = ReasoningBudgetState(maximumReasoningTokens: 2, endTokenIds: [10, 11])

        #expect(state.nextForcedTokenID == nil)
        #expect(state.didSample(1) == .counted(reasoningTokenCount: 1))
        #expect(state.nextForcedTokenID == nil)
        #expect(state.didSample(2) == .budgetReached(reasoningTokenCount: 2, nextTokenID: 10))
        #expect(state.nextForcedTokenID == 10)
        #expect(state.didSample(10) == .forcing(nextTokenID: 11))
        #expect(state.nextForcedTokenID == 11)
        #expect(state.didSample(11) == .closed(forced: true))
        #expect(state.nextForcedTokenID == nil)
        #expect(state.isClosed)
    }

    @Test("state stops when the model naturally emits the end marker")
    func stateStopsWhenModelNaturallyEmitsEndMarker() {
        var state = ReasoningBudgetState(maximumReasoningTokens: 10, endTokenIds: [10, 11])

        #expect(state.didSample(3) == .counted(reasoningTokenCount: 1))
        #expect(state.didSample(10) == .counted(reasoningTokenCount: 2))
        #expect(state.didSample(11) == .closed(forced: false))
        #expect(state.nextForcedTokenID == nil)
        #expect(state.isClosed)
    }

    @Test("generation parameters carry reasoning budget configuration")
    func generationParametersCarryReasoningBudgetConfiguration() async {
        let session = MLXSession()
        let sampling = SamplingParameters(
            temperature: 0.7,
            topP: 1.0,
            advanced: AdvancedSamplingParameters(
                reasoningBudget: ReasoningBudgetConfiguration(maximumTokens: 64)
            )
        )

        let parameters = await session.createGenerateParameters(
            from: sampling,
            limits: ResourceLimits(maxTokens: 8),
            reasoningEndTokenIds: [10, 11]
        )

        #expect(parameters.reasoningBudgetTokens == 64)
        #expect(parameters.reasoningEndTokenIds == [10, 11])
    }

    @Test("diagnostics include reasoning budget parameters")
    func diagnosticsIncludeReasoningBudgetParameters() async throws {
        let parameters = GenerateParameters(
            reasoningBudgetTokens: 64,
            reasoningEndTokenIds: [10, 11]
        )

        let recorded = try await MLXGenerationDiagnostics.withRecording {
            MLXGenerationDiagnostics.recordParameters(parameters)
        }
        let snapshot = try #require(recorded.events.compactMap { event in
            if case .parameters(let snapshot) = event {
                return snapshot
            }
            return nil
        }.first)

        #expect(snapshot.reasoningBudgetTokens == 64)
        #expect(snapshot.reasoningEndTokenCount == 2)
    }

    @Test(
        "processor masks logits to the forced reasoning end token",
        .disabled(
            if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
            "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
        )
    )
    func processorMasksLogitsToForcedReasoningEndToken() throws {
        let values = Device.withDefaultDevice(.cpu) {
            var processor = ReasoningBudgetProcessor(
                maximumReasoningTokens: 1,
                endTokenIds: [2]
            )
            processor.didSample(token: MLXArray(7))
            let logits = MLXArray([
                Float(0), Float(10), Float(20), Float(30)
            ]).reshaped([1, 4])

            let masked = processor.process(logits: logits)
            eval(masked)
            return masked.asArray(Float.self)
        }

        #expect(values == [-Float.infinity, -Float.infinity, 20, -Float.infinity])
    }
}
