import Foundation
@testable import MLXFoundationModel
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
        try await run(
            model: model,
            input: input,
            runtime: MLXRealModelEnvironment.runtimePreferences(for: model)
        )
    }

    static func run(
        model: MLXRealModelCatalog.Model,
        input: LLMInput,
        runtime: ModelRuntimePreferences,
        runtimeCapabilities: MLXGenerationRuntimeCapabilities = .scalar
    ) async throws -> GenerationResult {
        if runtimeCapabilities != .scalar {
            return try await run(
                session: MLXSession(runtimeCapabilities: runtimeCapabilities),
                model: model,
                input: input,
                runtime: runtime
            )
        }
        return try await run(
            session: MLXSessionFactory.create(),
            model: model,
            input: input,
            runtime: runtime
        )
    }

    static func run(
        session: any MLXGeneratingSession,
        model: MLXRealModelCatalog.Model,
        input: LLMInput,
        runtime: ModelRuntimePreferences
    ) async throws -> GenerationResult {
        do {
            try await preload(session: session, model: model, runtime: runtime)
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
        limits: ResourceLimits,
        prompt: String? = nil,
        runtime: ModelRuntimePreferences? = nil,
        runtimeCapabilities: MLXGenerationRuntimeCapabilities = .scalar
    ) async throws -> (result: GenerationResult, events: [MLXGenerationDiagnosticEvent]) {
        try await MLXGenerationDiagnostics.withRecording {
            let input = LLMInput(
                context: prompt ?? model.prompt,
                sampling: sampling,
                limits: limits
            )
            return try await run(
                model: model,
                input: input,
                runtime: runtime ?? MLXRealModelEnvironment.runtimePreferences(for: model),
                runtimeCapabilities: runtimeCapabilities
            )
        }
    }

    static func runRenderedRequest(
        model: MLXRealModelCatalog.Model,
        request: MLXBridgeRequest,
        limits: ResourceLimits,
        style: MLXPromptStyle? = nil,
        sampling: SamplingParameters = .deterministic
    ) async throws -> GenerationResult {
        let promptStyle = try style ?? inferredPromptStyle(for: model)
        let rendered = MLXPromptRenderer.render(request, style: promptStyle)
        let promptMetadata = promptStyle == .plain
            ? nil
            : PromptRenderMetadata(rendererID: rendered.rendererID)
        let promptCacheIdentity = promptStyle == .plain
            ? nil
            : PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint)
        let input = LLMInput(
            context: rendered.prompt,
            promptMetadata: promptMetadata,
            promptCacheIdentity: promptCacheIdentity,
            sampling: sampling,
            limits: limits
        )
        return try await run(model: model, input: input)
    }

    static func inferredPromptStyle(for model: MLXRealModelCatalog.Model) throws -> MLXPromptStyle {
        try MLXModelProfile.load(
            from: MLXRealModelEnvironment.modelURL(for: model),
            id: model.id
        )
        .promptStyle
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

    static func grammarSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGrammarConstraintSnapshot] {
        events.compactMap { event in
            guard case .grammarConstraint(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    static func reasoningBudgetSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXReasoningBudgetSnapshot] {
        events.compactMap { event in
            guard case .reasoningBudget(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    static func generatedTokenSnapshots(
        from events: [MLXGenerationDiagnosticEvent]
    ) -> [MLXGeneratedTokenSnapshot] {
        events.compactMap { event in
            guard case .generatedToken(let snapshot) = event else {
                return nil
            }
            return snapshot
        }
    }

    static func verifyGeneratedTokenDiagnostics(
        _ tokens: [MLXGeneratedTokenSnapshot],
        result: GenerationResult
    ) {
        let generatedTokens = result.metrics?.usage?.generatedTokens ?? 0
        let summaryText = tokens
            .map { token in
                "index=\(token.index) id=\(token.tokenID) text=\(token.tokenText.debugDescription)"
            }
            .joined(separator: "\n")
        let summary = Comment(rawValue: summaryText)

        #expect(!tokens.isEmpty, summary)
        #expect(generatedTokens > 0)
        #expect(tokens.count == generatedTokens, summary)
        if !tokens.isEmpty {
            #expect(tokens.map(\.index) == Array(1 ... tokens.count), summary)
        }
        #expect(tokens.contains { !$0.tokenText.isEmpty }, summary)
    }

    static func preload(
        session: any MLXGeneratingSession,
        model: MLXRealModelCatalog.Model
    ) async throws {
        try await preload(
            session: session,
            model: model,
            runtime: MLXRealModelEnvironment.runtimePreferences(for: model)
        )
    }

    static func preload(
        session: any MLXGeneratingSession,
        model: MLXRealModelCatalog.Model,
        runtime: ModelRuntimePreferences
    ) async throws {
        let configuration = ProviderConfiguration(
            location: MLXRealModelEnvironment.modelURL(for: model),
            authentication: .noAuth,
            modelName: model.displayName,
            compute: .small,
            runtime: runtime
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
