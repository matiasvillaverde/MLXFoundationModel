@testable import MLXFoundationModel
@testable import MLXLocalModels

extension MLXRealModelHarness {
    static func translateRenderedRequest(
        model: MLXRealModelCatalog.Model,
        request: MLXBridgeRequest,
        limits: ResourceLimits,
        style: MLXPromptStyle? = nil,
        sampling: SamplingParameters = .deterministic
    ) async throws -> [MLXTranslatedStreamEvent] {
        let promptStyle = try style ?? inferredPromptStyle(for: model)
        let input = renderedInput(
            request: request,
            style: promptStyle,
            sampling: sampling,
            limits: limits
        )
        return try await translate(
            model: model,
            input: input,
            tools: request.tools,
            promptStyle: promptStyle,
            reasoningStartsOpen: MLXPromptTemplateRenderer.generationStartsInReasoning(
                reasoningOptions: request.effectiveReasoningOptions,
                style: promptStyle
            )
        )
    }

    private static func renderedInput(
        request: MLXBridgeRequest,
        style: MLXPromptStyle,
        sampling: SamplingParameters,
        limits: ResourceLimits
    ) -> LLMInput {
        let rendered = MLXPromptRenderer.render(request, style: style)
        let promptMetadata = style == .plain
            ? nil
            : PromptRenderMetadata(rendererID: rendered.rendererID)
        let promptCacheIdentity = style == .plain
            ? nil
            : PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint)
        return LLMInput(
            context: rendered.prompt,
            promptMetadata: promptMetadata,
            promptCacheIdentity: promptCacheIdentity,
            sampling: sampling,
            limits: limits
        )
    }

    private static func translate(
        model: MLXRealModelCatalog.Model,
        input: LLMInput,
        tools: [MLXBridgeToolDefinition],
        promptStyle: MLXPromptStyle,
        reasoningStartsOpen: Bool
    ) async throws -> [MLXTranslatedStreamEvent] {
        let session = MLXSessionFactory.create()
        let sink = MLXRealModelRecordingStreamEventSink()
        do {
            try await preload(session: session, model: model)
            try await MLXStreamEventTranslator().translate(
                await session.stream(input),
                into: sink,
                tools: tools,
                promptStyle: promptStyle,
                reasoningStartsOpen: reasoningStartsOpen
            )
            await session.unload()
            return await sink.snapshot()
        } catch {
            await session.unload()
            throw error
        }
    }
}
