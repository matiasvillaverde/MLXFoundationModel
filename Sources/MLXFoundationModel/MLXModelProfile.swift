import Foundation

/// Metadata derived from a downloaded MLX model directory.
///
/// Profiles let callers expose model-specific capabilities and prompt defaults
/// without loading weights or depending on backend configuration keys.
public struct MLXModelProfile: Codable, Equatable, Hashable, Sendable {
    public let id: String?
    public let modelType: String?
    public let architectures: [String]
    public let contextLength: Int?
    public let vocabularySize: Int?
    public let quantizationBits: Int?
    public let isMixtureOfExperts: Bool
    public let hasVisionConfig: Bool
    public let hasNativeChatTemplate: Bool
    public let defaultReasoning: MLXModelReasoningDefault?
    public let promptStyle: MLXPromptStyle
    public let runtimeKind: MLXModelRuntimeKind
    public let capabilities: MLXModelCapabilities
    public let optimization: MLXModelOptimizationProfile?

    public var optimizationProfile: MLXModelOptimizationProfile {
        optimization ?? .empty
    }

    public var isVisionModel: Bool {
        hasVisionConfig || capabilities.vision
    }

    public var requiresVLMRuntime: Bool {
        runtimeKind == .vlm
    }

    public var hasReasoningToggle: Bool {
        defaultReasoning != nil
    }

    public var usesReasoningByDefault: Bool {
        defaultReasoning == .enabled
    }

    public init(
        id: String? = nil,
        modelType: String? = nil,
        architectures: [String] = [],
        contextLength: Int? = nil,
        vocabularySize: Int? = nil,
        quantizationBits: Int? = nil,
        isMixtureOfExperts: Bool = false,
        hasVisionConfig: Bool = false,
        hasNativeChatTemplate: Bool = false,
        defaultReasoning: MLXModelReasoningDefault? = nil,
        promptStyle: MLXPromptStyle = .plain,
        runtimeKind: MLXModelRuntimeKind = .text,
        capabilities: MLXModelCapabilities = .text,
        optimization: MLXModelOptimizationProfile? = nil
    ) {
        self.id = id
        self.modelType = modelType
        self.architectures = architectures
        self.contextLength = contextLength
        self.vocabularySize = vocabularySize
        self.quantizationBits = quantizationBits
        self.isMixtureOfExperts = isMixtureOfExperts
        self.hasVisionConfig = hasVisionConfig
        self.hasNativeChatTemplate = hasNativeChatTemplate
        self.defaultReasoning = defaultReasoning
        self.promptStyle = promptStyle
        self.runtimeKind = runtimeKind
        self.capabilities = capabilities
        self.optimization = optimization
    }

    public static func load(
        from modelDirectory: URL,
        id: String? = nil
    ) throws -> Self {
        try MLXModelProfileFactory.load(from: modelDirectory, id: id)
    }

    static func make(
        config: [String: Any],
        tokenizerConfig: [String: Any] = [:],
        id: String? = nil
    ) -> Self {
        MLXModelProfileFactory.make(config: config, tokenizerConfig: tokenizerConfig, id: id)
    }
}
