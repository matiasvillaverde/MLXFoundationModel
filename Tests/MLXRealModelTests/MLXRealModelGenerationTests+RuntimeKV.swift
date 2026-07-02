import Foundation
@testable import MLXLocalModels
import Testing

extension MLXRealModelGenerationTests {
    @Test("Qwen3 runs rotating and quantized KV cache options")
    func qwen3RunsRuntimeKVCacheOptions() async throws {
        let models = try MLXRealModelCatalog.load()
        guard let model = try MLXRealModelHarness.selectedModel("qwen3-0.6b-4bit", in: models) else {
            return
        }

        try await Self.verifyRotatingKVCacheGeneration(model: model)
        try await Self.verifyQuantizedKVCacheGeneration(model: model)
    }

    @Test("selected attention models run rotating and quantized KV cache options")
    func selectedAttentionModelsRunRuntimeKVCacheOptions() async throws {
        let models = try MLXRealModelCatalog.load()
        let selected = MLXRealModelEnvironment.selectedModels(from: models)
            .filter(Self.supportsRuntimeKVCacheOptions)
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
                try await Self.verifyRotatingKVCacheGeneration(model: model)
                try await Self.verifyQuantizedKVCacheGeneration(model: model)
            } catch {
                failures.append("\(model.id): \(error)")
            }
        }
        #expect(failures.isEmpty, Comment(rawValue: failures.joined(separator: "\n")))
    }

    private static func supportsRuntimeKVCacheOptions(_ model: MLXRealModelCatalog.Model) -> Bool {
        !runtimeKVCacheUnsupportedArchitectures.contains(model.architecture.lowercased())
    }

    private static let runtimeKVCacheUnsupportedArchitectures: Set<String> = [
        "mamba",
        "mamba2",
        "rwkv7"
    ]

    private static func verifyRotatingKVCacheGeneration(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let maxKVSize = 64
        let observed = try await Self.runWithCacheSnapshots(
            model: model,
            limits: ResourceLimits(
                maxTokens: 8,
                maxTime: .seconds(120),
                reusePromptCache: false,
                maxKVSize: maxKVSize,
                prefillStepSize: 64
            )
        )
        let summary = Comment(rawValue: Self.cacheSummary(observed.events))

        try Self.requireGenerated(observed.result, model: model, feature: "rotating KV cache")
        #expect(Self.cacheSnapshots(from: observed.events).contains { snapshot in
            snapshot.entries.contains { entry in
                entry.maxSize == maxKVSize && entry.typeName.contains("Rotating")
            }
        }, summary)
    }

    private static func verifyQuantizedKVCacheGeneration(
        model: MLXRealModelCatalog.Model
    ) async throws {
        let observed = try await Self.runWithCacheSnapshots(
            model: model,
            limits: ResourceLimits(
                maxTokens: 8,
                maxTime: .seconds(120),
                reusePromptCache: false,
                kvCacheBits: 4,
                quantizedKVStart: 0
            )
        )
        let conversions = Self.quantizedKVConversions(from: observed.events)
        let summary = Comment(rawValue: Self.cacheSummary(observed.events))

        try Self.requireGenerated(observed.result, model: model, feature: "quantized KV cache")
        #expect(conversions.contains { conversion in
            conversion.kvBits == 4 && conversion.convertedCount > 0
        }, summary)
    }

    private static func requireGenerated(
        _ result: MLXRealModelHarness.GenerationResult,
        model: MLXRealModelCatalog.Model,
        feature: String
    ) throws {
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let summary = Comment(rawValue: [
            "\(model.id) \(feature) produced no visible text",
            "textChunkCount=\(result.textChunkCount)",
            "generatedTokens=\(result.metrics?.usage?.generatedTokens ?? 0)",
            "stopReason=\(String(describing: result.metrics?.generation?.stopReason))"
        ].joined(separator: "\n"))

        try #require(!text.isEmpty, summary)
        try #require(result.textChunkCount > 0, summary)
        try #require((result.metrics?.usage?.generatedTokens ?? 0) > 0, summary)
        try #require(result.metrics?.generation?.stopReason != nil, summary)
    }

    private static func runWithCacheSnapshots(
        model: MLXRealModelCatalog.Model,
        limits: ResourceLimits
    ) async throws -> (result: MLXRealModelHarness.GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try await MLXGenerationDiagnostics.withCacheSnapshotRecording {
                try await MLXRealModelHarness.run(
                    model: model,
                    prompt: Self.kvCachePrompt,
                    sampling: .deterministic,
                    limits: limits
                )
            }
        }
    }

    private static var kvCachePrompt: String {
        let sentence = [
            "Runtime KV cache validation keeps local generation bounded",
            "observable and deterministic."
        ].joined(separator: " ")
        let body = Array(repeating: sentence, count: 2).joined(separator: " ")
        return "\(body)\nReply with two short words."
    }

    private static func cacheSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXCacheSnapshot] {
        events.compactMap { event in
            guard case .cacheSnapshot(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func quantizedKVConversions(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXQuantizedKVConversionSnapshot] {
        events.compactMap { event in
            guard case .quantizedKVConversion(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    private static func cacheSummary(_ events: [MLXGenerationDiagnosticEvent]) -> String {
        [
            "snapshots=\(Self.cacheSnapshots(from: events))",
            "quantizedKV=\(Self.quantizedKVConversions(from: events))"
        ].joined(separator: "\n")
    }
}
