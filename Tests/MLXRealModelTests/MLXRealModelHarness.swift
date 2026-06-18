import Foundation
import MLXFoundationModel
@testable import MLXLocalModels
import Testing

enum MLXRealModelHarness {
    struct GenerationResult: Sendable {
        let text: String
        let textChunkCount: Int
        let metrics: ChunkMetrics?
    }

    static func run(
        model: MLXRealModelCatalog.Model,
        prompt: String? = nil,
        sampling: SamplingParameters = .deterministic,
        limits: ResourceLimits? = nil
    ) async throws -> GenerationResult {
        let input = LLMInput(
            context: prompt ?? model.prompt,
            sampling: sampling,
            limits: limits ?? ResourceLimits(maxTokens: model.maxTokens, maxTime: .seconds(120))
        )
        return try await run(model: model, input: input)
    }

    static func run(
        model: MLXRealModelCatalog.Model,
        input: LLMInput
    ) async throws -> GenerationResult {
        let session = MLXSessionFactory.create()
        do {
            try await preload(session: session, model: model)
            let result = try await collectGeneration(from: await session.stream(input))
            await session.unload()
            return result
        } catch {
            await session.unload()
            throw error
        }
    }

    static func runWithDiagnostics(
        model: MLXRealModelCatalog.Model,
        sampling: SamplingParameters,
        limits: ResourceLimits
    ) async throws -> (result: GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            try await run(model: model, sampling: sampling, limits: limits)
        }
    }

    static func runRenderedRequest(
        model: MLXRealModelCatalog.Model,
        request: MLXBridgeRequest,
        limits: ResourceLimits,
        style: MLXPromptStyle = .plain,
        sampling: SamplingParameters = .deterministic
    ) async throws -> GenerationResult {
        let rendered = MLXPromptRenderer.render(request, style: style)
        let input = LLMInput(
            context: rendered.prompt,
            promptMetadata: style == .plain ? nil : PromptRenderMetadata(rendererID: rendered.rendererID),
            promptCacheIdentity: style == .plain ? nil : PromptCacheIdentity(
                stableFingerprint: rendered.cacheFingerprint
            ),
            sampling: sampling,
            limits: limits
        )
        return try await run(model: model, input: input)
    }

    static func requireModel(
        _ id: String,
        in models: [MLXRealModelCatalog.Model]
    ) throws -> MLXRealModelCatalog.Model {
        let model = try #require(models.first { $0.id == id })
        #expect(
            MLXRealModelEnvironment.hasModelFiles(for: model),
            Comment(rawValue: MLXRealModelEnvironment.missingModelsMessage([model]))
        )
        return model
    }

    static func verifyGenerated(
        _ result: GenerationResult,
        expectedTokens: [String] = []
    ) {
        let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(!text.isEmpty)
        #expect(result.textChunkCount > 0)
        #expect((result.metrics?.usage?.generatedTokens ?? 0) > 0)
        #expect(result.metrics?.generation?.stopReason != nil)
        guard !expectedTokens.isEmpty else {
            return
        }
        let lowercaseText = text.lowercased()
        let containsExpectedToken = expectedTokens.contains { token in
            lowercaseText.contains(token.lowercased())
        }
        #expect(containsExpectedToken || text.count > 2)
    }

    static func parameterSnapshot(
        from events: [MLXGenerationDiagnosticEvent]
    ) throws -> MLXGenerationParameterSnapshot {
        let snapshots: [MLXGenerationParameterSnapshot] = events.compactMap { event in
            guard case .parameters(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
        return try #require(snapshots.last)
    }

    private static func preload(
        session: any MLXGeneratingSession,
        model: MLXRealModelCatalog.Model
    ) async throws {
        let configuration = ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: model.displayName,
            compute: .small,
            runtime: ModelRuntimePreferences(promptCachePolicy: .memory)
        )
        let progress = await session.preload(configuration: configuration)
        for try await _ in progress {
            // Consume preload progress before generation.
        }
    }

    private static func collectGeneration(
        from stream: AsyncThrowingStream<LLMStreamChunk, Error>
    ) async throws -> GenerationResult {
        var text = ""
        var textChunkCount = 0
        var metrics: ChunkMetrics?
        for try await chunk in stream {
            if case .text = chunk.event {
                text += chunk.text
                if !chunk.text.isEmpty {
                    textChunkCount += 1
                }
            }
            metrics = chunk.metrics ?? metrics
        }
        return GenerationResult(text: text, textChunkCount: textChunkCount, metrics: metrics)
    }
}
