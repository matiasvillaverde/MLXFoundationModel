#if FOUNDATION_MODELS_PROVIDER_API && canImport(FoundationModels)
import FoundationModels

@available(macOS 27.0, iOS 27.0, visionOS 27.0, *)
enum FoundationModelsPendingChannelEvent: Sendable {
    case reasoning(LanguageModelExecutorGenerationChannel.Reasoning)
    case response(LanguageModelExecutorGenerationChannel.Response)
    case toolCalls(LanguageModelExecutorGenerationChannel.ToolCalls)

    func send(into channel: LanguageModelExecutorGenerationChannel) async {
        switch self {
        case .reasoning(let event):
            await channel.send(event)

        case .response(let event):
            await channel.send(event)

        case .toolCalls(let event):
            await channel.send(event)
        }
    }
}
#endif
