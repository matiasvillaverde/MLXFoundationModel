import CoreGraphics
import CryptoKit
import Foundation

/// The actor protocol implemented by local MLX text-generation sessions.
public protocol MLXGeneratingSession: Actor {
    /// Stream text generation for a fully rendered prompt.
    func stream(_ input: LLMInput) -> AsyncThrowingStream<LLMStreamChunk, Error>

    /// Stop the active generation, if any.
    nonisolated func stop()

    /// Preload a model into memory.
    func preload(configuration: ProviderConfiguration) -> AsyncThrowingStream<Progress, Error>

    /// Unload the active model and release memory.
    func unload() async
}

/// Backwards-compatible name used by the copied MLX runtime internals.
public typealias LLMSession = MLXGeneratingSession

/// A fully rendered local generation request.
public struct LLMInput: Sendable {
    /// Prompt text to tokenize and send to the model.
    public let context: String

    /// Optional renderer metadata. Non-nil metadata marks the prompt as already formatted.
    public let promptMetadata: PromptRenderMetadata?

    /// Optional cache identity used to validate prompt/KV cache reuse.
    public let promptCacheIdentity: PromptCacheIdentity?

    /// Image inputs. Current MLX text models reject non-empty image arrays.
    public let images: [CGImage]

    /// Video inputs. Current MLX text models reject non-empty video arrays.
    public let videoURLs: [URL]

    /// Sampling configuration.
    public let sampling: SamplingParameters

    /// Generation resource limits.
    public let limits: ResourceLimits

    public init(
        context: String,
        promptMetadata: PromptRenderMetadata? = nil,
        promptCacheIdentity: PromptCacheIdentity? = nil,
        images: [CGImage] = [],
        videoURLs: [URL] = [],
        sampling: SamplingParameters = .default,
        limits: ResourceLimits = .default
    ) {
        self.context = context
        self.promptMetadata = promptMetadata
        self.promptCacheIdentity = promptCacheIdentity
        self.images = images
        self.videoURLs = videoURLs
        self.sampling = sampling
        self.limits = limits
    }
}

/// A streamed text chunk from the local backend.
public struct LLMStreamChunk: Sendable {
    public let text: String
    public let metrics: ChunkMetrics?
    public let event: StreamEvent

    public init(text: String, event: StreamEvent, metrics: ChunkMetrics? = nil) {
        self.text = text
        self.metrics = metrics
        self.event = event
    }
}

/// Events emitted during local streaming.
public enum StreamEvent: Sendable {
    case text
    case metrics
    case finished
    case error(Error)
}

/// Common local provider errors.
public enum LLMError: Error, Sendable {
    case authenticationFailed(String)
    case rateLimitExceeded(retryAfter: Duration?)
    case modelNotFound(String)
    case invalidConfiguration(String)
    case networkError(Error)
    case providerError(code: String, message: String)
}

/// Local provider configuration.
public struct ProviderConfiguration: Sendable, Equatable, Hashable {
    public let location: URL
    public let authentication: Authentication
    public let modelName: String
    public let compute: ComputeConfiguration
    public let runtime: ModelRuntimePreferences?

    public init(
        location: URL,
        authentication: Authentication = .noAuth,
        modelName: String,
        compute: ComputeConfiguration = .large,
        runtime: ModelRuntimePreferences? = nil
    ) {
        self.location = location
        self.authentication = authentication
        self.modelName = modelName
        self.compute = compute
        self.runtime = runtime
    }
}

/// Authentication methods for local or remote-compatible providers.
public enum Authentication: Sendable, Equatable, Hashable {
    case noAuth
    case apiKey(String)
}

/// Compute-resource hints for local inference.
public struct ComputeConfiguration: Sendable, Equatable, Hashable {
    public let contextSize: Int
    public let batchSize: Int
    public let threadCount: Int

    public init(contextSize: Int, batchSize: Int, threadCount: Int) {
        self.contextSize = contextSize
        self.batchSize = batchSize
        self.threadCount = threadCount
    }

    public static let small = ComputeConfiguration(contextSize: 512, batchSize: 8, threadCount: 4)
    public static let medium = ComputeConfiguration(contextSize: 2_048, batchSize: 512, threadCount: 8)
    public static let large = ComputeConfiguration(contextSize: 4_096, batchSize: 1_024, threadCount: 16)
}

/// Supported token-level grammar constraint kinds.
public enum GrammarConstraintKind: String, Sendable, Equatable, Hashable {
    case builtinJSON
    case jsonSchema
    case ebnf
    case regex
    case choices
}

/// Optional token-level grammar constraints for structured generation.
public struct GrammarSamplingConfiguration: Sendable, Equatable, Hashable {
    public let kind: GrammarConstraintKind
    public let grammar: String
    public let root: String
    public let strict: Bool

    public init(
        kind: GrammarConstraintKind,
        grammar: String = "",
        root: String = "root",
        strict: Bool = true
    ) {
        self.kind = kind
        self.grammar = grammar
        self.root = root
        self.strict = strict
    }

    public init(grammar: String, root: String = "root") {
        self.init(kind: .ebnf, grammar: grammar, root: root)
    }

    public static func jsonSchema(_ schema: String, strict: Bool = true) -> Self {
        Self(kind: .jsonSchema, grammar: schema, strict: strict)
    }

    public static func json() -> Self {
        Self(kind: .builtinJSON)
    }

    public static func regex(_ regex: String) -> Self {
        Self(kind: .regex, grammar: regex)
    }

    public static func choices(_ choices: [String]) -> Self {
        precondition(!choices.isEmpty, "Grammar choices must not be empty")
        let alternatives = choices
            .map { #""\#(Self.escapedEBNFLiteral($0))""# }
            .joined(separator: " | ")
        return Self(kind: .choices, grammar: "root ::= \(alternatives)")
    }

    private static func escapedEBNFLiteral(_ value: String) -> String {
        value.reduce(into: "") { result, character in
            switch character {
            case "\"":
                result += #"\""#

            case "\\":
                result += #"\\"#

            case "\n":
                result += #"\n"#

            case "\r":
                result += #"\r"#

            case "\t":
                result += #"\t"#

            default:
                result.append(character)
            }
        }
    }
}

/// Mirostat sampler mode.
public enum MirostatSamplingVersion: Sendable, Equatable, Hashable {
    case v1
    case v2
}

/// Mirostat sampler configuration.
public struct MirostatSamplingConfiguration: Sendable, Equatable, Hashable {
    public let version: MirostatSamplingVersion
    public let tau: Float
    public let eta: Float
    public let learningTokens: Int32

    public init(
        version: MirostatSamplingVersion = .v2,
        tau: Float = 5.0,
        eta: Float = 0.1,
        learningTokens: Int32 = 100
    ) {
        self.version = version
        self.tau = tau
        self.eta = eta
        self.learningTokens = learningTokens
    }
}

/// Exclude Top Choices sampler configuration.
public struct XtcSamplingConfiguration: Sendable, Equatable, Hashable {
    public let probability: Float
    public let threshold: Float
    public let minKeep: Int

    public init(probability: Float, threshold: Float = 0.1, minKeep: Int = 1) {
        self.probability = probability
        self.threshold = threshold
        self.minKeep = minKeep
    }
}

/// DRY sampler configuration for discouraging repeated sequences.
public struct DrySamplingConfiguration: Sendable, Equatable, Hashable {
    public let multiplier: Float
    public let base: Float
    public let allowedLength: Int32
    public let penaltyLastTokens: Int32
    public let sequenceBreakers: [String]

    public init(
        multiplier: Float,
        base: Float = 1.75,
        allowedLength: Int32 = 2,
        penaltyLastTokens: Int32 = -1,
        sequenceBreakers: [String] = ["\n", ":", "\"", "*"]
    ) {
        self.multiplier = multiplier
        self.base = base
        self.allowedLength = allowedLength
        self.penaltyLastTokens = penaltyLastTokens
        self.sequenceBreakers = sequenceBreakers
    }
}

/// Adaptive-p sampler configuration.
public struct AdaptivePSamplingConfiguration: Sendable, Equatable, Hashable {
    public let target: Float
    public let decay: Float

    public init(target: Float, decay: Float = 0.9) {
        self.target = target
        self.decay = decay
    }
}

/// Advanced local sampler options.
public struct AdvancedSamplingParameters: Sendable, Equatable, Hashable {
    public let minP: Float?
    public let typicalP: Float?
    public let topNSigma: Float?
    public let grammar: GrammarSamplingConfiguration?
    public let mirostat: MirostatSamplingConfiguration?
    public let xtc: XtcSamplingConfiguration?
    public let dry: DrySamplingConfiguration?
    public let adaptiveP: AdaptivePSamplingConfiguration?
    public let logitBias: [Int32: Float]

    public static let disabled = AdvancedSamplingParameters()

    public init(
        minP: Float? = nil,
        typicalP: Float? = nil,
        topNSigma: Float? = nil,
        grammar: GrammarSamplingConfiguration? = nil,
        mirostat: MirostatSamplingConfiguration? = nil,
        xtc: XtcSamplingConfiguration? = nil,
        dry: DrySamplingConfiguration? = nil,
        adaptiveP: AdaptivePSamplingConfiguration? = nil,
        logitBias: [Int32: Float] = [:]
    ) {
        self.minP = minP
        self.typicalP = typicalP
        self.topNSigma = topNSigma
        self.grammar = grammar
        self.mirostat = mirostat
        self.xtc = xtc
        self.dry = dry
        self.adaptiveP = adaptiveP
        self.logitBias = logitBias
    }
}

/// Parameters controlling token sampling.
public struct SamplingParameters: Sendable, Equatable, Hashable {
    public let temperature: Float
    public let topP: Float
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let frequencyPenalty: Float?
    public let presencePenalty: Float?
    public let repetitionPenaltyRange: Int?
    public let seed: Int?
    public let stopSequences: [String]
    public let advanced: AdvancedSamplingParameters

    public static let `default` = SamplingParameters(
        temperature: 0.7,
        topP: 0.9
    )

    public static let deterministic = SamplingParameters(
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        seed: 42
    )

    public init(
        temperature: Float,
        topP: Float,
        topK: Int? = nil,
        repetitionPenalty: Float? = nil,
        frequencyPenalty: Float? = nil,
        presencePenalty: Float? = nil,
        repetitionPenaltyRange: Int? = nil,
        seed: Int? = nil,
        stopSequences: [String] = [],
        advanced: AdvancedSamplingParameters = .disabled
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.frequencyPenalty = frequencyPenalty
        self.presencePenalty = presencePenalty
        self.repetitionPenaltyRange = repetitionPenaltyRange
        self.seed = seed
        self.stopSequences = stopSequences
        self.advanced = advanced
    }
}

/// Hard generation limits.
public struct ResourceLimits: Sendable, Equatable, Hashable {
    public let maxTokens: Int
    public let maxTime: Duration?
    public let collectDetailedMetrics: Bool
    public let reusePromptCache: Bool
    public let maxPromptCacheBytes: Int?
    public let maxKVSize: Int?
    public let kvCacheBits: Int?
    public let kvCacheGroupSize: Int
    public let quantizedKVStart: Int
    public let prefillStepSize: Int

    public static let `default` = ResourceLimits(maxTokens: 2_048)

    public init(
        maxTokens: Int,
        maxTime: Duration? = nil,
        collectDetailedMetrics: Bool = false,
        reusePromptCache: Bool = true,
        maxPromptCacheBytes: Int? = 134_217_728,
        maxKVSize: Int? = nil,
        kvCacheBits: Int? = nil,
        kvCacheGroupSize: Int = 64,
        quantizedKVStart: Int = 0,
        prefillStepSize: Int = 512
    ) {
        self.maxTokens = maxTokens
        self.maxTime = maxTime
        self.collectDetailedMetrics = collectDetailedMetrics
        self.reusePromptCache = reusePromptCache
        self.maxPromptCacheBytes = maxPromptCacheBytes
        self.maxKVSize = maxKVSize
        self.kvCacheBits = kvCacheBits
        self.kvCacheGroupSize = kvCacheGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.prefillStepSize = prefillStepSize
    }
}

public enum ModelResidencyPreference: String, Codable, CaseIterable, Sendable {
    case warm
    case cold
}

public enum PromptCachePolicy: String, Codable, CaseIterable, Sendable {
    case off
    case memory
    case persistent
}

public enum SpeculativeDecodingMode: String, Codable, CaseIterable, Sendable {
    case off
    case sameModelDraft = "sameModelDraft"
}

/// Runtime hosting preferences for a local model.
public struct ModelRuntimePreferences: Codable, Equatable, Hashable, Sendable {
    public let residencyPreference: ModelResidencyPreference
    public let isPinned: Bool
    public let idleTTLSeconds: Int?
    public let promptCachePolicy: PromptCachePolicy
    public let promptCacheByteLimit: Int
    public let speculativeDecodingMode: SpeculativeDecodingMode
    public let speculativeDraftTokens: Int

    public static let `default` = ModelRuntimePreferences(
        residencyPreference: .warm,
        isPinned: false,
        idleTTLSeconds: 300,
        promptCachePolicy: .persistent,
        promptCacheByteLimit: 134_217_728,
        speculativeDecodingMode: .off,
        speculativeDraftTokens: 2
    )

    public init(
        residencyPreference: ModelResidencyPreference = .warm,
        isPinned: Bool = false,
        idleTTLSeconds: Int? = 300,
        promptCachePolicy: PromptCachePolicy = .memory,
        promptCacheByteLimit: Int = 134_217_728,
        speculativeDecodingMode: SpeculativeDecodingMode = .off,
        speculativeDraftTokens: Int = 2
    ) {
        self.residencyPreference = residencyPreference
        self.isPinned = isPinned
        self.idleTTLSeconds = idleTTLSeconds
        self.promptCachePolicy = promptCachePolicy
        self.promptCacheByteLimit = promptCacheByteLimit
        self.speculativeDecodingMode = speculativeDecodingMode
        self.speculativeDraftTokens = speculativeDraftTokens
    }
}

/// Opaque prompt-render metadata for cache identity.
public struct PromptRenderMetadata: Codable, Sendable, Hashable {
    public let rendererID: String

    public init(rendererID: String) {
        self.rendererID = rendererID
    }
}

/// Stable cache identity for prompt token and backend KV cache reuse.
public struct PromptCacheIdentity: Codable, Sendable, Hashable {
    public let stableFingerprint: String

    public init(stableFingerprint: String) {
        self.stableFingerprint = stableFingerprint
    }

    public static func stableFingerprint(for value: String) -> String {
        let digest = SHA256.hash(data: Data(value.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }
}

/// Performance and usage metrics for a stream chunk.
public struct ChunkMetrics: Sendable, Codable {
    public let timing: TimingMetrics?
    public let usage: UsageMetrics?
    public let generation: GenerationMetrics?

    public init(
        timing: TimingMetrics? = nil,
        usage: UsageMetrics? = nil,
        generation: GenerationMetrics? = nil
    ) {
        self.timing = timing
        self.usage = usage
        self.generation = generation
    }
}

public struct TimingMetrics: Sendable, Codable {
    public let totalTime: Duration
    public let timeToFirstToken: Duration?
    public let timeSinceLastToken: Duration?
    public let tokenTimings: [Duration]
    public let promptProcessingTime: Duration?

    public init(
        totalTime: Duration,
        timeToFirstToken: Duration? = nil,
        timeSinceLastToken: Duration? = nil,
        tokenTimings: [Duration] = [],
        promptProcessingTime: Duration? = nil
    ) {
        self.totalTime = totalTime
        self.timeToFirstToken = timeToFirstToken
        self.timeSinceLastToken = timeSinceLastToken
        self.tokenTimings = tokenTimings
        self.promptProcessingTime = promptProcessingTime
    }
}

public struct UsageMetrics: Sendable, Codable {
    public let generatedTokens: Int
    public let totalTokens: Int
    public let promptTokens: Int?
    public let contextWindowSize: Int?
    public let contextTokensUsed: Int?
    public let kvCacheBytes: Int64?
    public let kvCacheEntries: Int?
    public let promptCacheReusedTokenCount: Int?

    public init(
        generatedTokens: Int,
        totalTokens: Int,
        promptTokens: Int? = nil,
        contextWindowSize: Int? = nil,
        contextTokensUsed: Int? = nil,
        kvCacheBytes: Int64? = nil,
        kvCacheEntries: Int? = nil,
        promptCacheReusedTokenCount: Int? = nil
    ) {
        self.generatedTokens = generatedTokens
        self.totalTokens = totalTokens
        self.promptTokens = promptTokens
        self.contextWindowSize = contextWindowSize
        self.contextTokensUsed = contextTokensUsed
        self.kvCacheBytes = kvCacheBytes
        self.kvCacheEntries = kvCacheEntries
        self.promptCacheReusedTokenCount = promptCacheReusedTokenCount
    }
}

public struct GenerationMetrics: Sendable, Codable {
    public enum StopReason: String, Sendable, Codable {
        case maxTokens = "max_tokens"
        case stopSequence = "stop_sequence"
        case endOfSequence = "end_of_sequence"
        case userRequested = "user_requested"
        case timeout
        case error
    }

    public let stopReason: StopReason?
    public let temperature: Float32?
    public let topP: Float32?
    public let topK: Int32?

    public init(
        stopReason: StopReason? = nil,
        temperature: Float32? = nil,
        topP: Float32? = nil,
        topK: Int32? = nil
    ) {
        self.stopReason = stopReason
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }
}
