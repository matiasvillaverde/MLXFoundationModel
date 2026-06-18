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
        let bridge = try bridgeRequest(from: request)
        let rendered = MLXPromptRenderer.render(bridge, style: model.model.promptStyle)
        let maxTokens = request.generationOptions.maximumResponseTokens ?? model.maximumResponseTokens
        return LLMInput(
            context: rendered.prompt,
            promptMetadata: PromptRenderMetadata(rendererID: rendered.rendererID),
            promptCacheIdentity: PromptCacheIdentity(stableFingerprint: rendered.cacheFingerprint),
            sampling: sampling(from: request.generationOptions, fallback: model.sampling),
            limits: ResourceLimits(maxTokens: maxTokens)
        )
    }

    private static func bridgeRequest(
        from request: LanguageModelExecutorGenerationRequest
    ) throws -> MLXBridgeRequest {
        var instructions: [String] = []
        var messages: [MLXBridgeMessage] = []
        for entry in request.transcript {
            append(entry, to: &messages, instructions: &instructions)
        }
        return MLXBridgeRequest(
            messages: messages,
            instructions: instructions.filter { !$0.isEmpty }.joined(separator: "\n\n"),
            tools: request.enabledToolDefinitions.map(toolDefinition)
        )
    }

    private static func append(
        _ entry: Transcript.Entry,
        to messages: inout [MLXBridgeMessage],
        instructions: inout [String]
    ) {
        switch entry {
        case .instructions(let value):
            instructions.append(text(of: value.segments))

        case .prompt(let value):
            messages.append(MLXBridgeMessage(role: .user, content: text(of: value.segments)))

        case .response(let value):
            messages.append(MLXBridgeMessage(role: .assistant, content: text(of: value.segments)))

        case .toolCalls(let calls):
            messages.append(
                MLXBridgeMessage(
                    role: .assistant,
                    content: toolCallsText(calls)
                )
            )

        case .toolOutput(let output):
            messages.append(
                MLXBridgeMessage(
                    role: .tool,
                    content: text(of: output.segments),
                    name: output.id
                )
            )

        case .reasoning:
            return

        @unknown default:
            return
        }
    }

    private static func text(of segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text):
                text.content

            case .structure(let structure):
                structure.content.jsonString

            case .attachment, .custom:
                nil

            @unknown default:
                nil
            }
        }
        .joined(separator: "\n")
    }

    private static func toolDefinition(
        _ definition: Transcript.ToolDefinition
    ) -> MLXBridgeToolDefinition {
        MLXBridgeToolDefinition(
            name: definition.name,
            description: definition.description,
            parametersJSONSchema: jsonSchemaString(from: definition.parameters)
        )
    }

    private static func jsonSchemaString(from schema: GenerationSchema) -> String {
        guard
            let data = try? JSONEncoder().encode(schema),
            let string = String(data: data, encoding: .utf8)
        else {
            return "{}"
        }
        return string
    }

    private static func toolCallsText(_ calls: Transcript.ToolCalls) -> String {
        calls.map { call in
            """
            {"tool_name":"\(call.toolName)","arguments":\(call.arguments.jsonString)}
            """
        }
        .joined(separator: "\n")
    }

    private static func sampling(
        from options: GenerationOptions,
        fallback: SamplingParameters
    ) -> SamplingParameters {
        var temperature = options.temperature.map(Float.init) ?? fallback.temperature
        var topP = fallback.topP
        var topK = fallback.topK

        switch options.samplingMode?.kind {
        case .greedy:
            temperature = 0
            topK = 1

        case .top(let value, _):
            topK = value

        case .nucleus(let threshold, _):
            topP = Float(threshold)

        case nil:
            break

        @unknown default:
            break
        }

        return SamplingParameters(
            temperature: temperature,
            topP: topP,
            topK: topK,
            repetitionPenalty: fallback.repetitionPenalty,
            frequencyPenalty: fallback.frequencyPenalty,
            presencePenalty: fallback.presencePenalty,
            repetitionPenaltyRange: fallback.repetitionPenaltyRange,
            seed: fallback.seed,
            stopSequences: fallback.stopSequences,
            advanced: fallback.advanced
        )
    }
}
#endif
