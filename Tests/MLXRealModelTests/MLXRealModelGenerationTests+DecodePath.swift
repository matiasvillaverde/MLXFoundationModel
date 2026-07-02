import MLXFoundationModel
@testable import MLXLocalModels
import Testing

extension MLXRealModelGenerationTests {
    @Test("selected models report greedy and constrained decode paths")
    func selectedModelsReportGreedyAndConstrainedDecodePaths() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
        let missing = selected.filter { !MLXRealModelEnvironment.hasModelFiles(for: $0) }

        #expect(!selected.isEmpty)
        #expect(
            missing.isEmpty,
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage(missing))
        )
        guard missing.isEmpty else {
            return
        }

        var failures: [String] = []
        for model in selected {
            do {
                try await Self.verifyGreedyDecodePath(model: model)
                try await Self.verifyConstrainedDecodePath(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    @Test("Qwen3 reports greedy and constrained decode paths")
    func qwen3ReportsGreedyAndConstrainedDecodePaths() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        try await Self.verifyGreedyDecodePath(model: model)
        try await Self.verifyConstrainedDecodePath(model: model)
    }

    private static func verifyGreedyDecodePath(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 4, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "/no_think\nReply with one short word."
        )
        let snapshots = MLXRealModelHarness.decodePathSnapshots(from: observed.events)
        let summary = Comment(rawValue: Self.decodePathSummary(snapshots))
        let allUseArgmaxSampler = snapshots.allSatisfy(\.argmaxSampler)
        let allHaveGreedyModel = snapshots.allSatisfy(\.greedyModelAvailable)

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(!snapshots.isEmpty, summary)
        #expect(snapshots.allSatisfy { $0.path == .greedyToken }, summary)
        #expect(snapshots.allSatisfy { !$0.processorActive }, summary)
        #expect(allUseArgmaxSampler, summary)
        #expect(allHaveGreedyModel, summary)
    }

    private static func verifyConstrainedDecodePath(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await MLXRealModelHarness.runWithDiagnostics(
            model: model,
            sampling: Self.singleTokenJSONSampling,
            limits: ResourceLimits(maxTokens: 1, maxTime: .seconds(120), reusePromptCache: false),
            prompt: "/no_think\nDo not output JSON. Say hello in plain English."
        )
        let snapshots = MLXRealModelHarness.decodePathSnapshots(from: observed.events)
        let summary = Comment(rawValue: Self.decodePathSummary(snapshots))
        let allUseProcessor = snapshots.allSatisfy(\.processorActive)
        let allUseArgmaxSampler = snapshots.allSatisfy(\.argmaxSampler)
        let allHaveGreedyModel = snapshots.allSatisfy(\.greedyModelAvailable)

        MLXRealModelHarness.verifyGenerated(observed.result)
        #expect(!snapshots.isEmpty, summary)
        #expect(snapshots.allSatisfy { $0.path == .logits }, summary)
        #expect(allUseProcessor, summary)
        #expect(allUseArgmaxSampler, summary)
        #expect(allHaveGreedyModel, summary)
    }

    private static var singleTokenJSONSampling: SamplingParameters {
        SamplingParameters(
            temperature: 0,
            topP: 1,
            topK: 1,
            seed: 42,
            advanced: AdvancedSamplingParameters(
                grammar: GrammarSamplingConfiguration(grammar: #"root ::= "{""#)
            )
        )
    }

    private static func decodePathSummary(_ snapshots: [MLXDecodePathSnapshot]) -> String {
        let lines = snapshots.map { snapshot in
            """
            path=\(snapshot.path.rawValue) processor=\(snapshot.processorActive) \
            argmax=\(snapshot.argmaxSampler) greedyModel=\(snapshot.greedyModelAvailable)
            """
        }
        return lines.joined(separator: "\n")
    }
}
