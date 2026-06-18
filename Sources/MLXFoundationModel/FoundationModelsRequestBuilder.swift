#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsRequestBuilder {
    static func build(
        from request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws -> LLMInput {
        let bridge = try bridgeRequest(from: request, model: model)
        let rendered = MLXPromptRenderer.render(bridge, style: model.model.promptStyle)
        let maxTokens = request.generationOptions.maximumResponseTokens ?? model.maximumResponseTokens
        let promptMetadata = promptMetadata(for: rendered, style: model.model.promptStyle)
        let cacheIdentity = promptCacheIdentity(for: rendered, style: model.model.promptStyle)
        let toolDefinitions = effectiveToolDefinitions(from: request)
        let sampling = FoundationModelsSamplingMapper.sampling(
            from: request.generationOptions,
            schema: request.schema,
            toolDefinitions: toolDefinitions,
            fallback: model.sampling
        )
        return LLMInput(
            context: rendered.prompt,
            promptMetadata: promptMetadata,
            promptCacheIdentity: cacheIdentity,
            sampling: sampling,
            limits: ResourceLimits(maxTokens: maxTokens)
        )
    }

    private static func promptMetadata(
        for rendered: MLXRenderedRequest,
        style: MLXPromptStyle
    ) -> PromptRenderMetadata? {
        guard style != .plain else {
            return nil
        }
        return PromptRenderMetadata(rendererID: rendered.rendererID)
    }

    private static func promptCacheIdentity(
        for rendered: MLXRenderedRequest,
        style: MLXPromptStyle
    ) -> PromptCacheIdentity? {
        guard style != .plain else {
            return nil
        }
        return PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint)
    }

    private static func bridgeRequest(
        from request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws -> MLXBridgeRequest {
        var converted = FoundationModelsTranscriptBridge.convert(request.transcript)
        guard converted.unsupportedEntries.isEmpty else {
            throw LanguageModelError.unsupportedTranscriptContent(.init(
                unsupportedContent: converted.unsupportedEntries,
                debugDescription: "MLXFoundationModel only accepts text and structured transcript segments",
                metadata: [
                    "model": model.model.id
                ]
            ))
        }
        let toolDefinitions = effectiveToolDefinitions(from: request)
        let requiresToolCall = FoundationModelsSamplingMapper.requiresToolCall(request.generationOptions)
        if requiresToolCall, !toolDefinitions.isEmpty {
            converted.instructions.append("Call one of the available tools before producing a final answer.")
        }
        return MLXBridgeRequest(
            messages: converted.messages,
            instructions: converted.instructions.filter { !$0.isEmpty }.joined(separator: "\n\n"),
            responseConstraint: responseConstraint(from: request),
            tools: toolDefinitions.map(FoundationModelsToolSchemaBuilder.bridgeToolDefinition)
        )
    }

    private static func responseConstraint(
        from schema: GenerationSchema
    ) -> MLXBridgeResponseConstraint {
        MLXBridgeResponseConstraint(
            jsonSchema: FoundationModelsSchemaSupport.jsonSchemaString(from: schema),
            instructions: "Return only a structured response matching this schema."
        )
    }

    private static func responseConstraint(
        from request: LanguageModelExecutorGenerationRequest
    ) -> MLXBridgeResponseConstraint? {
        guard request.contextOptions.includeSchemaInPrompt ?? true else {
            return nil
        }
        return request.schema.map(responseConstraint)
    }

    private static func effectiveToolDefinitions(
        from request: LanguageModelExecutorGenerationRequest
    ) -> [Transcript.ToolDefinition] {
        switch request.generationOptions.toolCallingMode?.kind {
        case .disallowed:
            return []

        case .allowed, .required, nil:
            return request.enabledToolDefinitions

        @unknown default:
            return request.enabledToolDefinitions
        }
    }
}
#endif
