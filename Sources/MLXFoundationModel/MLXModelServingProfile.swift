import Foundation
import MLXLocalModels

/// API-visible serving profile for a registered model.
///
/// A serving profile exposes a derived model identifier such as
/// `qwen:deterministic` while reusing the same downloaded model directory.
/// Runtime changes may still require a separate resident engine, but
/// sampling-only profiles share the loaded weights.
public struct MLXModelServingProfile: Equatable, Hashable, Sendable {
    public let name: String
    public let aliases: [String]
    public let runtime: ModelRuntimePreferences?
    public let sampling: SamplingParameters?
    public let maximumResponseTokens: Int?

    public init(
        name: String,
        aliases: [String] = [],
        runtime: ModelRuntimePreferences? = nil,
        sampling: SamplingParameters? = nil,
        maximumResponseTokens: Int? = nil
    ) {
        self.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        self.aliases = aliases.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        self.runtime = runtime
        self.sampling = sampling
        self.maximumResponseTokens = maximumResponseTokens.map { max(1, $0) }
    }

    func applying(
        to model: MLXLanguageModel,
        publicID: String
    ) -> MLXLanguageModel {
        MLXLanguageModel(
            model: MLXModel(
                id: publicID,
                location: model.model.location,
                promptStyle: model.model.promptStyle,
                capabilities: model.model.capabilities,
                profile: model.model.profile
            ),
            compute: model.compute,
            runtime: runtime ?? model.runtime,
            sampling: sampling ?? model.sampling,
            maximumResponseTokens: maximumResponseTokens ?? model.maximumResponseTokens
        )
    }
}
