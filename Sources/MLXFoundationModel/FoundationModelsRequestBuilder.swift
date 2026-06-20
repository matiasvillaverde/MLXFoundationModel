#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import CoreGraphics
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsRequestBuilder {
    private struct BridgeBuildResult {
        let request: MLXBridgeRequest
        let images: [CGImage]
    }

    static func build(
        from request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws -> LLMInput {
        let bridge = try bridgeRequest(from: request, model: model)
        let rendered = MLXPromptRenderer.render(bridge.request, style: model.model.promptStyle)
        let maxTokens = request.generationOptions.maximumResponseTokens ?? model.maximumResponseTokens
        let promptMetadata = promptMetadata(for: rendered, style: model.model.promptStyle)
        let cacheIdentity = promptCacheIdentity(for: rendered, style: model.model.promptStyle)
        let toolDefinitions = effectiveToolDefinitions(from: request)
        let sampling = FoundationModelsSamplingMapper.sampling(
            from: request,
            toolDefinitions: toolDefinitions,
            promptStyle: model.model.promptStyle,
            fallback: model.sampling
        )
        return LLMInput(
            context: rendered.prompt,
            promptMetadata: promptMetadata,
            promptCacheIdentity: cacheIdentity,
            images: bridge.images,
            sampling: sampling,
            limits: ResourceLimits(maxTokens: maxTokens)
        )
    }

    static func bridgeToolDefinitions(
        from request: LanguageModelExecutorGenerationRequest
    ) -> [MLXBridgeToolDefinition] {
        effectiveToolDefinitions(from: request).map(FoundationModelsToolSchemaBuilder.bridgeToolDefinition)
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
    ) throws -> BridgeBuildResult {
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
        try validateCapabilities(
            request: request,
            model: model,
            converted: converted,
            toolDefinitions: toolDefinitions
        )
        appendRequestInstructions(to: &converted, request: request, toolDefinitions: toolDefinitions)
        let bridgeRequest = makeBridgeRequest(
            converted: converted,
            request: request,
            model: model,
            toolDefinitions: toolDefinitions
        )
        return BridgeBuildResult(request: bridgeRequest, images: converted.images)
    }

    private static func appendRequestInstructions(
        to converted: inout FoundationModelsTranscriptBridge.Result,
        request: LanguageModelExecutorGenerationRequest,
        toolDefinitions: [Transcript.ToolDefinition]
    ) {
        if let reasoningInstruction = reasoningInstruction(for: request.contextOptions.reasoningLevel) {
            converted.instructions.append(reasoningInstruction)
        }
        let requiresToolCall = FoundationModelsSamplingMapper.requiresToolCall(request.generationOptions)
        if requiresToolCall, !toolDefinitions.isEmpty {
            converted.instructions.append("Call one of the available tools before producing a final answer.")
        }
    }

    private static func makeBridgeRequest(
        converted: FoundationModelsTranscriptBridge.Result,
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        toolDefinitions: [Transcript.ToolDefinition]
    ) -> MLXBridgeRequest {
        let reasoningOptions = reasoningOptions(
            for: request.contextOptions.reasoningLevel,
            model: model
        )
        return MLXBridgeRequest(
            messages: converted.messages,
            instructions: converted.instructions.filter { !$0.isEmpty }.joined(separator: "\n\n"),
            reasoningEnabled: reasoningOptions?.isEnabled ?? false,
            reasoningOptions: reasoningOptions,
            responseConstraint: responseConstraint(from: request),
            tools: toolDefinitions.map(FoundationModelsToolSchemaBuilder.bridgeToolDefinition)
        )
    }

    private static func validateCapabilities(
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        converted: FoundationModelsTranscriptBridge.Result,
        toolDefinitions: [Transcript.ToolDefinition]
    ) throws {
        try validateSchemaCapability(request: request, model: model)
        try validateToolCapability(request: request, model: model, toolDefinitions: toolDefinitions)
        try validateVisionCapability(converted: converted, model: model)
        try validateReasoningCapability(request: request, model: model)
    }

    private static func validateSchemaCapability(
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws {
        if request.schema != nil, !model.model.capabilities.structuredOutput {
            throw unsupportedCapability(
                .guidedGeneration,
                model: model.model.id,
                reason: "Guided generation was requested with a schema, but the model does not advertise it."
            )
        }
    }

    private static func validateToolCapability(
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel,
        toolDefinitions: [Transcript.ToolDefinition]
    ) throws {
        if !toolDefinitions.isEmpty, !model.model.capabilities.toolCalling {
            throw unsupportedCapability(
                .toolCalling,
                model: model.model.id,
                reason: "Tool definitions were enabled, but the model does not advertise tool calling."
            )
        }
        let requiredWithoutTools = FoundationModelsSamplingMapper.requiresToolCall(request.generationOptions)
            && toolDefinitions.isEmpty
        if requiredWithoutTools {
            throw unsupportedCapability(
                .toolCalling,
                model: model.model.id,
                reason: "Tool calling was required, but no tools are enabled for this request."
            )
        }
    }

    private static func validateVisionCapability(
        converted: FoundationModelsTranscriptBridge.Result,
        model: MLXLanguageModel
    ) throws {
        if converted.containsImageAttachments, !model.model.capabilities.vision {
            throw unsupportedCapability(
                .vision,
                model: model.model.id,
                reason: "Image attachments were present, but the model does not advertise vision."
            )
        }
        if converted.containsImageAttachments, !model.supportsVisionExecution {
            throw unsupportedCapability(
                .vision,
                model: model.model.id,
                reason: "Image attachments require a VLM runtime, which this executor does not provide yet."
            )
        }
    }

    private static func validateReasoningCapability(
        request: LanguageModelExecutorGenerationRequest,
        model: MLXLanguageModel
    ) throws {
        if request.contextOptions.reasoningLevel != nil, !model.model.capabilities.reasoning {
            throw unsupportedCapability(
                .reasoning,
                model: model.model.id,
                reason: "A reasoning level was requested, but the model does not advertise reasoning."
            )
        }
    }

    private static func unsupportedCapability(
        _ capability: LanguageModelCapabilities.Capability,
        model: String,
        reason: String
    ) -> LanguageModelError {
        LanguageModelError.unsupportedCapability(.init(
            capability: capability,
            debugDescription: "\(reason) model=\(model)",
            metadata: ["model": model]
        ))
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
