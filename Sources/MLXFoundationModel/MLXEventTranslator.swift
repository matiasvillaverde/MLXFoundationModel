#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import Foundation
import FoundationModels
import MLXLocalModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
struct MLXEventTranslator: Sendable {
    private struct TranslationOutput {
        var text = ""
        var latestMetrics: ChunkMetrics?
    }

    private let responseEntryID = UUID().uuidString
    private let toolCallsEntryID = UUID().uuidString

    func translate(
        _ stream: AsyncThrowingStream<LLMStreamChunk, Error>,
        into channel: LanguageModelExecutorGenerationChannel,
        toolDefinitionsEnabled: Bool
    ) async throws {
        let output = try await collect(
            stream,
            into: channel,
            toolDefinitionsEnabled: toolDefinitionsEnabled
        )
        if toolDefinitionsEnabled {
            await finishToolAware(output, into: channel)
        } else {
            await sendResponseUsageIfAvailable(output.latestMetrics?.usage, into: channel)
        }
    }

    private func collect(
        _ stream: AsyncThrowingStream<LLMStreamChunk, Error>,
        into channel: LanguageModelExecutorGenerationChannel,
        toolDefinitionsEnabled: Bool
    ) async throws -> TranslationOutput {
        var output = TranslationOutput()
        for try await chunk in stream {
            try Task.checkCancellation()
            output.latestMetrics = chunk.metrics ?? output.latestMetrics
            guard !chunk.text.isEmpty else {
                continue
            }
            output.text += chunk.text
            if !toolDefinitionsEnabled {
                await sendResponseText(chunk.text, tokenCount: 1, into: channel)
            }
        }
        return output
    }

    private func finishToolAware(
        _ output: TranslationOutput,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        guard let call = MLXToolCallExtractor.extract(from: output.text) else {
            await flushBufferedResponse(output, into: channel)
            return
        }
        await sendToolCall(call, into: channel)
        await sendToolUsageIfAvailable(output.latestMetrics?.usage, into: channel)
    }

    private func flushBufferedResponse(
        _ output: TranslationOutput,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        if !output.text.isEmpty {
            await sendResponseText(
                output.text,
                tokenCount: output.latestMetrics?.usage?.generatedTokens ?? 1,
                into: channel
            )
        }
        await sendResponseUsageIfAvailable(output.latestMetrics?.usage, into: channel)
    }

    private func sendToolCall(
        _ call: MLXExtractedToolCall,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await channel.send(
            .toolCalls(
                entryID: toolCallsEntryID,
                action: .toolCall(
                    id: UUID().uuidString,
                    name: call.name,
                    action: .appendArguments(call.argumentsJSON, tokenCount: 1)
                )
            )
        )
    }

    private func sendResponseText(
        _ text: String,
        tokenCount: Int,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        await channel.send(
            .response(
                entryID: responseEntryID,
                action: .appendText(text, tokenCount: max(tokenCount, 1))
            )
        )
    }

    private func sendResponseUsageIfAvailable(
        _ usage: UsageMetrics?,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        guard let usage else {
            return
        }
        await channel.send(
            .response(
                entryID: responseEntryID,
                action: .updateUsage(
                    input: inputUsage(from: usage),
                    output: outputUsage(from: usage)
                )
            )
        )
    }

    private func sendToolUsageIfAvailable(
        _ usage: UsageMetrics?,
        into channel: LanguageModelExecutorGenerationChannel
    ) async {
        guard let usage else {
            return
        }
        await channel.send(
            .toolCalls(
                entryID: toolCallsEntryID,
                action: .updateUsage(
                    input: inputUsage(from: usage),
                    output: outputUsage(from: usage)
                )
            )
        )
    }

    private func inputUsage(
        from usage: UsageMetrics
    ) -> LanguageModelExecutorGenerationChannel.Usage.Input {
        LanguageModelExecutorGenerationChannel.Usage.Input(
            totalTokenCount: usage.promptTokens ?? max(usage.totalTokens - usage.generatedTokens, 0),
            cachedTokenCount: usage.promptCacheReusedTokenCount ?? 0
        )
    }

    private func outputUsage(
        from usage: UsageMetrics
    ) -> LanguageModelExecutorGenerationChannel.Usage.Output {
        LanguageModelExecutorGenerationChannel.Usage.Output(
            totalTokenCount: usage.generatedTokens,
            reasoningTokenCount: 0
        )
    }
}
#endif
