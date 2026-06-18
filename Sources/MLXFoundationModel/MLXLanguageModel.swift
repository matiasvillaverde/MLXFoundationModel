import Foundation
@_exported import MLXLocalModels

/// MLX-backed language model entry point for Foundation Models sessions.
///
/// When Apple's `FoundationModels` provider APIs are present, this type conforms
/// to `LanguageModel` in `MLXFoundationModelsConformance.swift`.
public struct MLXLanguageModel: Equatable, Hashable, Sendable {
    public let model: MLXModel
    public let compute: ComputeConfiguration
    public let runtime: ModelRuntimePreferences
    public let sampling: SamplingParameters
    public let maximumResponseTokens: Int

    public init(
        model: MLXModel,
        compute: ComputeConfiguration = .large,
        runtime: ModelRuntimePreferences = .default,
        sampling: SamplingParameters = .default,
        maximumResponseTokens: Int = 2_048
    ) {
        self.model = model
        self.compute = compute
        self.runtime = runtime
        self.sampling = sampling
        self.maximumResponseTokens = maximumResponseTokens
    }

    public var providerConfiguration: ProviderConfiguration {
        ProviderConfiguration(
            location: model.location,
            authentication: .noAuth,
            modelName: model.id,
            compute: compute,
            runtime: runtime
        )
    }
}
