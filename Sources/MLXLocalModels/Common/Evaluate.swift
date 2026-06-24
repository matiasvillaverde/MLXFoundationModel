import Foundation
@preconcurrency import MLX
import MLXNN
import Tokenizers

/// Namespace for text generation entry points.
internal enum TextGenerator {}

/// Selects one token from a model logits row.
internal protocol LogitSampler: Sendable {

    func sample(logits: MLXArray) -> MLXArray
}

/// Stateful hook that can observe prompts, adjust logits, and track sampled tokens.
internal protocol LogitProcessor: Sendable {

    mutating func prompt(_ prompt: MLXArray)

    mutating func process(logits: MLXArray) -> MLXArray

    mutating func didSample(token: MLXArray)
}

/// User and runtime controls for token generation.
internal struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    internal var prefillStepSize: Int

    /// Alignment policy for partial prompt-cache reuse.
    internal var promptCacheReuseAlignment: PromptCacheReuseAlignment

    /// Maximum tokens to generate
    internal var maxTokens: Int?

    /// Maximum size of the key-value cache. Old entries (except the first N tokens) will be overwritten,
    /// where N is determined by ``GenerationConstants/rotatingCacheKeepTokens``.
    /// When set, uses ``RotatingKVCache`` instead of ``KVCacheSimple``
    internal var maxKVSize: Int?

    /// Number of bits to use for KV cache quantization. nil implies no cache quantization.
    internal var kvBits: Int?

    /// Group size for KV cache quantization (default: 64)
    internal var kvGroupSize: Int = GenerationConstants.defaultKVCacheGroupSize

    /// Step to begin using a quantized KV cache when kvBits is non-nil (default: 0)
    internal var quantizedKVStart: Int = 0

    /// Whether dynamic KV quantization should leave the last model layer in full precision.
    internal var quantizedKVSkipLastLayer: Bool = false

    /// Every Nth DeepSeek DSA layer computes indexer top-k indices. nil disables IndexCache.
    internal var indexCacheFrequency: Int?

    /// sampling temperature
    internal var temperature: Float = 0.6

    /// top p sampling
    internal var topP: Float = 1.0

    /// Top-k sampling. A value of 0 disables this filter.
    internal var topK: Int = 0

    /// Min-p sampling threshold relative to the highest-probability token.
    internal var minP: Float = 0.0

    /// Locally typical sampling probability mass. A value of 1 disables the filter.
    internal var typicalP: Float = 1.0

    /// Top-n-sigma logit threshold. nil disables the filter.
    internal var topNSigma: Float?

    /// Exclude Top Choices probability. A value of 0 disables XTC filtering.
    internal var xtcProbability: Float = 0.0

    /// Exclude Top Choices probability threshold.
    internal var xtcThreshold: Float = 0.1

    /// Minimum number of above-threshold candidates XTC must keep.
    internal var xtcMinKeep: Int = 1

    /// Tokens that must never be removed by XTC filtering.
    internal var xtcProtectedTokenIds: Set<Int>

    /// Mirostat adaptive surprise sampler. nil disables Mirostat.
    internal var mirostat: MirostatSamplingConfiguration?

    /// DRY token-level repeat-sequence penalty. nil disables DRY.
    internal var dry: DrySamplingConfiguration?

    /// Tokenized DRY sequence breakers, each in chronological token order.
    internal var drySequenceBreakerTokenIds: [[Int]]

    /// Adaptive-p target-probability sampler. nil disables adaptive-p.
    internal var adaptiveP: AdaptivePSamplingConfiguration?

    /// penalty factor for repeating tokens
    internal var repetitionPenalty: Float?

    /// number of tokens to consider for repetition penalty
    internal var repetitionContextSize: Int = GenerationConstants.defaultRepetitionContextSize

    /// Additive penalty for tokens that appear in recent context.
    internal var presencePenalty: Float?

    /// Number of tokens to consider for presence penalty.
    internal var presenceContextSize: Int = GenerationConstants.defaultRepetitionContextSize

    /// Additive penalty that scales with token frequency in recent context.
    internal var frequencyPenalty: Float?

    /// Number of tokens to consider for frequency penalty.
    internal var frequencyContextSize: Int = GenerationConstants.defaultRepetitionContextSize

    /// Optional random seed for reproducible stochastic sampling.
    internal var seed: Int?

    /// Per-token additive logit bias.
    internal var logitBias: [Int: Float]

    /// Maximum reasoning tokens before forcing the configured reasoning end marker.
    internal var reasoningBudgetTokens: Int?

    /// Token sequence that closes the model's active reasoning section.
    internal var reasoningEndTokenIds: [Int]

    /// Token ids loaded from model generation config that must never be sampled.
    internal var suppressTokenIds: Set<Int>

    /// Token-level grammar constraint.
    internal var grammar: GrammarSamplingConfiguration?

    internal init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = GenerationConstants.defaultKVCacheGroupSize,
        quantizedKVStart: Int = 0,
        quantizedKVSkipLastLayer: Bool = false,
        indexCacheFrequency: Int? = nil,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        typicalP: Float? = nil,
        topNSigma: Float? = nil,
        xtcProbability: Float = 0.0,
        xtcThreshold: Float = 0.1,
        xtcMinKeep: Int = 1,
        xtcProtectedTokenIds: Set<Int> = [],
        mirostat: MirostatSamplingConfiguration? = nil,
        dry: DrySamplingConfiguration? = nil,
        drySequenceBreakerTokenIds: [[Int]] = [],
        adaptiveP: AdaptivePSamplingConfiguration? = nil,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        seed: Int? = nil,
        grammar: GrammarSamplingConfiguration? = nil,
        logitBias: [Int: Float] = [:],
        reasoningBudgetTokens: Int? = nil,
        reasoningEndTokenIds: [Int] = [],
        suppressTokenIds: Set<Int> = [],
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize,
        promptCacheReuseAlignment: PromptCacheReuseAlignment = .exact
    ) {
        self.maxTokens = maxTokens.map { max(0, $0) }
        self.maxKVSize = maxKVSize.map { max(1, $0) }
        self.kvBits = kvBits
        self.kvGroupSize = max(1, kvGroupSize)
        self.quantizedKVStart = quantizedKVStart
        self.quantizedKVSkipLastLayer = quantizedKVSkipLastLayer
        self.indexCacheFrequency = indexCacheFrequency
        self.temperature = temperature
        self.topP = topP
        self.topK = max(0, topK)
        self.minP = minP
        self.typicalP = typicalP.map(Self.normalizedProbability) ?? 1.0
        self.topNSigma = Self.normalizedTopNSigma(topNSigma)
        self.xtcProbability = Self.normalizedProbability(xtcProbability)
        self.xtcThreshold = Self.normalizedXTCThreshold(xtcThreshold)
        self.xtcMinKeep = max(1, xtcMinKeep)
        self.xtcProtectedTokenIds = Self.normalizedTokenIdSet(xtcProtectedTokenIds)
        self.mirostat = Self.normalizedMirostat(mirostat)
        self.dry = Self.normalizedDry(dry)
        self.drySequenceBreakerTokenIds = Self.normalizedSequenceBreakers(
            drySequenceBreakerTokenIds
        )
        self.adaptiveP = Self.normalizedAdaptiveP(adaptiveP)
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.seed = seed
        self.grammar = grammar
        self.logitBias = Self.normalizedLogitBias(logitBias)
        self.reasoningBudgetTokens = reasoningBudgetTokens.map { max(1, $0) }
        self.reasoningEndTokenIds = Self.normalizedTokenIds(reasoningEndTokenIds)
        self.suppressTokenIds = Self.normalizedTokenIdSet(suppressTokenIds)
        self.prefillStepSize = max(1, prefillStepSize)
        self.promptCacheReuseAlignment = promptCacheReuseAlignment
    }

    private static func normalizedProbability(_ probability: Float) -> Float {
        min(max(probability, 0), 1)
    }

    private static func normalizedXTCThreshold(_ threshold: Float) -> Float {
        min(max(threshold, 0), 0.5)
    }

    private static func normalizedTopNSigma(_ value: Float?) -> Float? {
        guard let value, value > 0 else { return nil }
        return value
    }

    private static func normalizedDry(
        _ value: DrySamplingConfiguration?
    ) -> DrySamplingConfiguration? {
        guard let value,
              value.multiplier != 0,
              value.base >= 1,
              value.allowedLength >= 0,
              value.penaltyLastTokens != 0
        else {
            return nil
        }
        return value
    }

    private static func normalizedMirostat(
        _ value: MirostatSamplingConfiguration?
    ) -> MirostatSamplingConfiguration? {
        guard let value, value.tau > 0, value.eta > 0, value.learningTokens > 1 else {
            return nil
        }
        return value
    }

    private static func normalizedAdaptiveP(
        _ value: AdaptivePSamplingConfiguration?
    ) -> AdaptivePSamplingConfiguration? {
        guard let value, value.target.isFinite, value.decay.isFinite, value.target >= 0 else {
            return nil
        }
        return AdaptivePSamplingConfiguration(
            target: min(value.target, 1),
            decay: min(max(value.decay, 0), 0.99)
        )
    }

    private static func normalizedSequenceBreakers(_ values: [[Int]]) -> [[Int]] {
        values.compactMap { sequence in
            let tokens = sequence.filter { $0 >= 0 }
            return tokens.isEmpty ? nil : tokens
        }
    }

    private static func normalizedTokenIds(_ values: [Int]) -> [Int] {
        values.filter { $0 >= 0 }
    }

    private static func normalizedTokenIdSet(_ values: Set<Int>) -> Set<Int> {
        Set(values.filter { $0 >= 0 })
    }

    private static func normalizedLogitBias(_ values: [Int: Float]) -> [Int: Float] {
        values.filter { key, _ in key >= 0 }
    }

    internal var usesMirostatSampler: Bool {
        samplerPlan.usesMirostatSampler
    }

    internal var usesAdaptivePSampler: Bool {
        samplerPlan.usesAdaptivePSampler
    }

    internal func sampler() -> LogitSampler {
        samplerPlan.makeSampler()
    }

    internal func processor(
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws -> LogitProcessor? {
        try LogitProcessorPlan(parameters: self, grammarCompiler: grammarCompiler)
            .makeProcessor()
    }

    private var samplerPlan: LogitSamplerPlan {
        LogitSamplerPlan(parameters: self)
    }
}

private struct LogitSamplerPlan {
    let temperature: Float
    let topP: Float
    let topK: Int
    let minP: Float
    let typicalP: Float
    let topNSigma: Float?
    let xtcProbability: Float
    let xtcThreshold: Float
    let xtcMinKeep: Int
    let xtcProtectedTokenIds: Set<Int>
    let mirostat: MirostatSamplingConfiguration?
    let adaptiveP: AdaptivePSamplingConfiguration?
    let seed: Int?

    init(parameters: GenerateParameters) {
        self.temperature = parameters.temperature
        self.topP = parameters.topP
        self.topK = parameters.topK
        self.minP = parameters.minP
        self.typicalP = parameters.typicalP
        self.topNSigma = parameters.topNSigma
        self.xtcProbability = parameters.xtcProbability
        self.xtcThreshold = parameters.xtcThreshold
        self.xtcMinKeep = parameters.xtcMinKeep
        self.xtcProtectedTokenIds = parameters.xtcProtectedTokenIds
        self.mirostat = parameters.mirostat
        self.adaptiveP = parameters.adaptiveP
        self.seed = parameters.seed
    }

    var usesMirostatSampler: Bool {
        mirostat != nil && temperature > 0
    }

    var usesAdaptivePSampler: Bool {
        adaptiveP != nil && temperature > 0
    }

    private var usesProbabilityFilters: Bool {
        (topP > 0 && topP < 1)
            || topK > 0
            || minP > 0
            || (typicalP > 0 && typicalP < 1)
            || topNSigma != nil
            || xtcProbability > 0
    }

    func makeSampler() -> LogitSampler {
        if temperature == 0 {
            return ArgMaxSampler()
        }

        if let mirostat, usesMirostatSampler {
            return MirostatSampler(configuration: mirostat, temperature: temperature, seed: seed)
        }

        if let adaptiveP, usesAdaptivePSampler {
            return AdaptivePSampler(
                configuration: adaptiveP,
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                typicalP: typicalP,
                topNSigma: topNSigma,
                xtcProbability: xtcProbability,
                xtcThreshold: xtcThreshold,
                xtcMinKeep: xtcMinKeep,
                xtcProtectedTokenIds: xtcProtectedTokenIds,
                seed: seed
            )
        }

        if usesProbabilityFilters {
            return TopPSampler(
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                typicalP: typicalP,
                topNSigma: topNSigma,
                xtcProbability: xtcProbability,
                xtcThreshold: xtcThreshold,
                xtcMinKeep: xtcMinKeep,
                xtcProtectedTokenIds: xtcProtectedTokenIds,
                seed: seed
            )
        }

        return CategoricalSampler(temperature: temperature, seed: seed)
    }
}

private struct LogitProcessorPlan {
    var logitBiasContext: LogitBiasProcessor?
    var suppressTokenContext: SuppressTokensProcessor?
    var reasoningBudgetContext: ReasoningBudgetProcessor?
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?
    var dryContext: DryPenaltyContext?
    var grammarContext: GrammarConstrainedLogitProcessor?

    init(
        parameters: GenerateParameters,
        grammarCompiler: GrammarConstraintCompiler?
    ) throws {
        self.logitBiasContext = parameters.logitBias.isEmpty
            ? nil
            : LogitBiasProcessor(logitBias: parameters.logitBias)
        self.suppressTokenContext = parameters.suppressTokenIds.isEmpty
            ? nil
            : SuppressTokensProcessor(tokenIds: parameters.suppressTokenIds)
        self.reasoningBudgetContext = Self.reasoningBudgetContext(parameters)
        self.repetitionContext = Self.repetitionContext(parameters)
        self.presenceContext = Self.presenceContext(parameters)
        self.frequencyContext = Self.frequencyContext(parameters)
        self.dryContext = Self.dryContext(parameters)
        self.grammarContext = try Self.grammarContext(parameters, compiler: grammarCompiler)
    }

    var isEmpty: Bool {
        logitBiasContext == nil
            && suppressTokenContext == nil
            && reasoningBudgetContext == nil
            && repetitionContext == nil
            && presenceContext == nil
            && frequencyContext == nil
            && dryContext == nil
            && grammarContext == nil
    }

    func makeProcessor() -> LogitProcessor? {
        guard !isEmpty else { return nil }
        return PenaltyProcessor(
            logitBiasContext: logitBiasContext,
            suppressTokenContext: suppressTokenContext,
            reasoningBudgetContext: reasoningBudgetContext,
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext,
            dryContext: dryContext,
            grammarContext: grammarContext
        )
    }

    private static func repetitionContext(_ parameters: GenerateParameters) -> RepetitionContext? {
        guard let penalty = parameters.repetitionPenalty,
              penalty != 0,
              penalty != 1,
              parameters.repetitionContextSize > 0
        else {
            return nil
        }
        return RepetitionContext(
            repetitionPenalty: penalty,
            repetitionContextSize: parameters.repetitionContextSize
        )
    }

    private static func presenceContext(
        _ parameters: GenerateParameters
    ) -> PresencePenaltyContext? {
        guard let penalty = parameters.presencePenalty,
              penalty != 0,
              parameters.presenceContextSize > 0
        else {
            return nil
        }
        return PresencePenaltyContext(
            presencePenalty: penalty,
            presenceContextSize: parameters.presenceContextSize
        )
    }

    private static func frequencyContext(
        _ parameters: GenerateParameters
    ) -> FrequencyPenaltyContext? {
        guard let penalty = parameters.frequencyPenalty,
              penalty != 0,
              parameters.frequencyContextSize > 0
        else {
            return nil
        }
        return FrequencyPenaltyContext(
            frequencyPenalty: penalty,
            frequencyContextSize: parameters.frequencyContextSize
        )
    }

    private static func dryContext(_ parameters: GenerateParameters) -> DryPenaltyContext? {
        guard let dry = parameters.dry else { return nil }
        return DryPenaltyContext(
            configuration: dry,
            totalContextSize: max(parameters.maxKVSize ?? Int.max, 1),
            sequenceBreakers: parameters.drySequenceBreakerTokenIds
        )
    }

    private static func reasoningBudgetContext(
        _ parameters: GenerateParameters
    ) -> ReasoningBudgetProcessor? {
        guard let budget = parameters.reasoningBudgetTokens,
              budget > 0,
              !parameters.reasoningEndTokenIds.isEmpty
        else {
            return nil
        }
        return ReasoningBudgetProcessor(
            maximumReasoningTokens: budget,
            endTokenIds: parameters.reasoningEndTokenIds
        )
    }

    private static func grammarContext(
        _ parameters: GenerateParameters,
        compiler: GrammarConstraintCompiler?
    ) throws -> GrammarConstrainedLogitProcessor? {
        guard let grammar = parameters.grammar else {
            return nil
        }
        guard let compiler else {
            throw GrammarConstraintError.missingGrammarCompiler
        }
        return try GrammarConstrainedLogitProcessor(
            matcher: compiler.makeMatcher(for: grammar)
        )
    }
}

/// Sampler that uses `argMax` (most likely) to sample the logits.
internal struct ArgMaxSampler: LogitSampler {
    internal func sample(logits: MLX.MLXArray) -> MLX.MLXArray {
        argMax(logits, axis: -1)
    }
}

/// Sampler that uses probability filters (`topP`, `topK`, `minP`) and `temperature`.
///
/// This type is marked `@unchecked Sendable` because:
/// - It contains `MLXArray` instances which are NOT inherently Sendable
/// - It contains `MLXRandom.RandomState` which is NOT inherently Sendable
/// - However, each sampler instance is used in isolation during generation
/// - The arrays are immutable after creation (`let` constants)
/// - The random state is only accessed from a single thread context (within `generate`)
///
/// Safety guarantees:
/// - Immutable configuration: filter values are never modified after creation
/// - Single-threaded usage: Used within `generate` function which runs in ModelContainer context
/// - Isolated random state: Each sampler has its own random state, no sharing
/// - No escaping: Sampler is created and used within a single generation cycle
private struct ProbabilityFilterPipeline: @unchecked Sendable {
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let typicalP: MLXArray?
    let topNSigma: MLXArray?
    let xtcProbability: MLXArray?
    let xtcThreshold: MLXArray
    let xtcMinKeep: Int
    let xtcProtectedTokenIds: [Int]
    let posInf: MLXArray
    let negInf: MLXArray

    init(
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        typicalP: Float = 1.0,
        topNSigma: Float? = nil,
        xtcProbability: Float = 0.0,
        xtcThreshold: Float = 0.1,
        xtcMinKeep: Int = 1,
        xtcProtectedTokenIds: Set<Int> = []
    ) {
        self.topP = topP > 0 && topP < 1 ? MLXArray(topP) : nil
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.typicalP = typicalP > 0 && typicalP < 1 ? MLXArray(typicalP) : nil
        self.topNSigma = topNSigma.map { MLXArray(Float($0)) }
        self.xtcProbability = xtcProbability > 0 ? MLXArray(xtcProbability) : nil
        self.xtcThreshold = MLXArray(xtcThreshold)
        self.xtcMinKeep = max(1, xtcMinKeep)
        self.xtcProtectedTokenIds = xtcProtectedTokenIds
            .filter { $0 >= 0 }
            .sorted()
        self.posInf = MLXArray(Float.infinity)
        self.negInf = MLXArray(-Float.infinity)
    }

    var isActive: Bool {
        topP != nil || topK != nil || minP != nil || typicalP != nil || topNSigma != nil
            || xtcProbability != nil
    }

    func apply(to logprobs: MLXArray, logits: MLXArray) -> MLXArray {
        var logprobs = logprobs

        if let topNSigma {
            logprobs = applyTopNSigma(logprobs, logits: logits, nSigma: topNSigma)
        }
        if let topP {
            logprobs = applyTopP(logprobs, topP: topP)
        }
        if let typicalP {
            logprobs = applyTypicalP(logprobs, typicalP: typicalP)
        }
        if let minP {
            logprobs = applyMinP(logprobs, minP: minP)
        }
        if let xtcProbability {
            logprobs = applyXTC(logprobs, probability: xtcProbability)
        }
        if let topK {
            logprobs = applyTopK(logprobs, topK: topK)
        }

        return logprobs
    }

    private func applyTopP(_ logprobs: MLXArray, topP: MLXArray) -> MLXArray {
        let sortedIndices = argSort(logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let sortedProbs = exp(sortedLogprobs)
        let cumulativeProbs = cumsum(sortedProbs, axis: -1)
        let filtered = MLX.where(cumulativeProbs .> (1 - topP), sortedLogprobs, negInf)

        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private func applyMinP(_ logprobs: MLXArray, minP: MLXArray) -> MLXArray {
        let maxLogprob = logprobs.max(axis: -1, keepDims: true)
        let threshold = maxLogprob + log(minP)

        return MLX.where(logprobs .>= threshold, logprobs, negInf)
    }

    private func applyTypicalP(_ logprobs: MLXArray, typicalP: MLXArray) -> MLXArray {
        let probs = exp(logprobs)
        let entropy = -(probs * logprobs).sum(axis: -1, keepDims: true)
        let shiftedScores = abs(-logprobs - entropy)
        let sortedIndices = argSort(shiftedScores, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let cumulativeProbs = cumsum(exp(sortedLogprobs), axis: -1)
        let positions = MLXArray.arange(logprobs.dim(-1)).reshaped(1, -1)
        let keepMask = MLX.logicalOr(cumulativeProbs .<= typicalP, positions .== 0)
        let filtered = MLX.where(keepMask, sortedLogprobs, negInf)

        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private func applyTopNSigma(
        _ logprobs: MLXArray,
        logits: MLXArray,
        nSigma: MLXArray
    ) -> MLXArray {
        let threshold = logits.max(axis: -1, keepDims: true) -
            nSigma * std(logits, axis: -1, keepDims: true)
        return MLX.where(logits .>= threshold, logprobs, negInf)
    }

    private func applyXTC(_ logprobs: MLXArray, probability: MLXArray) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        let probs = exp(logprobs)
        let maskableProbs: MLXArray
        if let protectedMask = protectedTokenMask(vocabularySize: vocabularySize) {
            maskableProbs = MLX.where(protectedMask .> 0, MLXArray.zeros(like: probs), probs)
        } else {
            maskableProbs = probs
        }

        let candidates = MLX.where(maskableProbs .> xtcThreshold, maskableProbs, posInf)
        let cutoff = xtcCutoff(candidates, vocabularySize: vocabularySize)
        let tokenMask = maskableProbs .> cutoff
        let batchSize = logprobs.dim(0)
        let shouldApply = MLXRandom.uniform(low: Float(0), high: Float(1), [batchSize, 1]) .< probability
        let filtered = MLX.where(tokenMask, negInf, logprobs)

        return MLX.where(shouldApply, filtered, logprobs)
    }

    private func protectedTokenMask(vocabularySize: Int) -> MLXArray? {
        let tokenIds = xtcProtectedTokenIds.filter { $0 < vocabularySize }
        guard !tokenIds.isEmpty else { return nil }

        let indices = MLXArray(tokenIds).asType(.int32)
        let values = MLXArray.ones([tokenIds.count], type: Float32.self)
        return MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[indices]
            .add(values)
            .reshaped(1, -1)
    }

    private func xtcCutoff(_ candidates: MLXArray, vocabularySize: Int) -> MLXArray {
        if xtcMinKeep <= 1 {
            return candidates.min(axis: -1, keepDims: true)
        }

        let keepIndex = min(xtcMinKeep - 1, vocabularySize - 1)
        return sorted(candidates, axis: -1)[0..., keepIndex].reshaped(-1, 1)
    }

    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }

        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
    }
}

internal struct TopPSampler: LogitSampler, @unchecked Sendable {
    let temp: MLXArray
    private let filters: ProbabilityFilterPipeline
    let randomState: MLXRandom.RandomState

    internal init(
        temperature: Float,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        typicalP: Float = 1.0,
        topNSigma: Float? = nil,
        xtcProbability: Float = 0.0,
        xtcThreshold: Float = 0.1,
        xtcMinKeep: Int = 1,
        xtcProtectedTokenIds: Set<Int> = [],
        seed: Int? = nil
    ) {
        self.temp = MLXArray(temperature)
        self.filters = ProbabilityFilterPipeline(
            topP: topP,
            topK: topK,
            minP: minP,
            typicalP: typicalP,
            topNSigma: topNSigma,
            xtcProbability: xtcProbability,
            xtcThreshold: xtcThreshold,
            xtcMinKeep: xtcMinKeep,
            xtcProtectedTokenIds: xtcProtectedTokenIds
        )
        self.randomState = makeRandomState(seed: seed)
    }

    internal func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            logprobs = filters.apply(to: logprobs, logits: logits)

            return categorical(logprobs * (1 / temp))
        }
    }
}

/// Processor that uses `temperature` to sample the logits
///
/// This type is marked `@unchecked Sendable` because:
/// - It contains `MLXArray` which is NOT inherently Sendable
/// - It contains `MLXRandom.RandomState` which is NOT inherently Sendable
/// - However, each sampler instance is used in isolation during generation
/// - The array is immutable after creation (`let` constant)
/// - The random state is only accessed from a single thread context (within `generate`)
///
/// Safety guarantees:
/// - Immutable configuration: `temp` is never modified after creation
/// - Single-threaded usage: Used within `generate` function which runs in ModelContainer context
/// - Isolated random state: Each sampler has its own random state, no sharing
/// - No escaping: Sampler is created and used within a single generation cycle
internal struct CategoricalSampler: LogitSampler, @unchecked Sendable {
    let temp: MLXArray
    let randomState: MLXRandom.RandomState

    internal init(temperature: Float, seed: Int? = nil) {
        self.temp = MLXArray(temperature)
        self.randomState = makeRandomState(seed: seed)
    }

    internal func sample(logits: MLXArray) -> MLXArray {
        return withRandomState(randomState) {
            categorical(logits * (1 / temp))
        }
    }
}

/// Stateful adaptive-p sampler.
///
/// This mirrors llama.cpp's adaptive-p sampler: existing probability filters are applied first,
/// then logits are transformed toward an adaptive target probability before sampling. Only the
/// selected token's original probability is read back to update the EMA state.
internal final class AdaptivePSampler: LogitSampler, @unchecked Sendable {
    private static let distributionWidth: Float = 0.3
    private static let peakLogitValue: Float = 5.0
    private static let sharpness: Float = 10.0
    private static let inverseWidth: Float = 1.0 / distributionWidth

    private let target: Float
    private let decay: Float
    private let temp: MLXArray
    private let filters: ProbabilityFilterPipeline
    private let negInf = MLXArray(-Float.infinity)
    private let randomState: MLXRandom.RandomState
    private var weightedSum: Float
    private var totalWeight: Float

    internal init(
        configuration: AdaptivePSamplingConfiguration,
        temperature: Float,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        typicalP: Float = 1.0,
        topNSigma: Float? = nil,
        xtcProbability: Float = 0.0,
        xtcThreshold: Float = 0.1,
        xtcMinKeep: Int = 1,
        xtcProtectedTokenIds: Set<Int> = [],
        seed: Int? = nil
    ) {
        self.target = min(max(configuration.target, 0), 1)
        self.decay = min(max(configuration.decay, 0), 0.99)
        self.temp = MLXArray(temperature)
        self.filters = ProbabilityFilterPipeline(
            topP: topP,
            topK: topK,
            minP: minP,
            typicalP: typicalP,
            topNSigma: topNSigma,
            xtcProbability: xtcProbability,
            xtcThreshold: xtcThreshold,
            xtcMinKeep: xtcMinKeep,
            xtcProtectedTokenIds: xtcProtectedTokenIds
        )
        self.randomState = makeRandomState(seed: seed)
        self.weightedSum = self.target / (1.0 - self.decay)
        self.totalWeight = 1.0 / (1.0 - self.decay)
    }

    internal func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)
            logprobs = filters.apply(to: logprobs, logits: logits)
            logprobs = logprobs * (1 / temp)
            let originalProbs = exp(logSoftmax(logprobs))
            let transformedLogits = adaptivePLogits(originalProbs: originalProbs, logprobs: logprobs)
            let transformedLogprobs = logSoftmax(transformedLogits)
            let sampled = categorical(transformedLogprobs)
            updateState(sampledToken: sampled, originalProbs: originalProbs)
            return sampled
        }
    }

    private func adaptivePLogits(originalProbs: MLXArray, logprobs: MLXArray) -> MLXArray {
        let adaptedTarget = max(
            min(
                totalWeight == 0 ? target : 2.0 * target - (weightedSum / totalWeight),
                1.0
            ),
            0.0
        )
        let distance = abs((originalProbs - MLXArray(adaptedTarget)) * Self.inverseWidth)
        let logits = MLXArray(Self.peakLogitValue) -
            MLXArray(Self.sharpness) * distance * distance / (1.0 + distance)

        return MLX.where(isFinite(logprobs), logits, negInf)
    }

    private func updateState(sampledToken: MLXArray, originalProbs: MLXArray) {
        eval(sampledToken)
        let tokenID = sampledToken.item(Int.self)
        let selectedIndex = MLXArray([Int32(tokenID)]).reshaped(1, 1)
        let selectedProbability = takeAlong(originalProbs, selectedIndex, axis: -1)
        eval(selectedProbability)
        let probability = selectedProbability.item(Float.self)
        weightedSum = probability + decay * weightedSum
        totalWeight = 1.0 + decay * totalWeight
    }
}

/// Stateful Mirostat sampler with adaptive surprise feedback.
///
/// The filtering work stays in MLX arrays; only the selected token probability is read back to
/// update the scalar feedback state for the next token.
internal final class MirostatSampler: LogitSampler, @unchecked Sendable {
    private static let log2 = Float(log(2.0))
    private static let epsilon: Float = 1e-6

    private let version: MirostatSamplingVersion
    private let tau: Float
    private let eta: Float
    private let learningTokens: Int
    private let temp: MLXArray
    private let negInf = MLXArray(-Float.infinity)
    private let randomState: MLXRandom.RandomState
    private var mu: Float

    internal init(
        configuration: MirostatSamplingConfiguration,
        temperature: Float,
        seed: Int? = nil
    ) {
        self.version = configuration.version
        self.tau = configuration.tau
        self.eta = configuration.eta
        self.learningTokens = max(2, Int(configuration.learningTokens))
        self.temp = MLXArray(temperature)
        self.randomState = makeRandomState(seed: seed)
        self.mu = 2 * configuration.tau
    }

    internal func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            let logprobs = logSoftmax(logits * (1 / temp))
            let filtered = switch version {
            case .v1:
                applyMirostatV1(logprobs)

            case .v2:
                applyMirostatV2(logprobs)
            }
            let truncatedLogprobs = logSoftmax(filtered)
            let sampled = categorical(truncatedLogprobs)
            updateState(sampledToken: sampled, logprobs: truncatedLogprobs)
            return sampled
        }
    }

    private func applyMirostatV1(_ logprobs: MLXArray) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        let sortedIndices = argSort(-logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let topK = mirostatV1TopK(sortedLogprobs: sortedLogprobs, vocabularySize: vocabularySize)
        let positions = MLXArray.arange(vocabularySize).reshaped(1, -1)
        let keepMask = positions .< topK
        let filtered = MLX.where(keepMask, sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private func applyMirostatV2(_ logprobs: MLXArray) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        let sortedIndices = argSort(-logprobs, axis: -1)
        let sortedLogprobs = takeAlong(logprobs, sortedIndices, axis: -1)
        let surprise = -sortedLogprobs / Self.log2
        let positions = MLXArray.arange(vocabularySize).reshaped(1, -1)
        let keepMask = MLX.logicalOr(surprise .<= MLXArray(mu), positions .== 0)
        let filtered = MLX.where(keepMask, sortedLogprobs, negInf)
        return putAlong(logprobs, sortedIndices, values: filtered, axis: -1)
    }

    private func mirostatV1TopK(
        sortedLogprobs: MLXArray,
        vocabularySize: Int
    ) -> Int {
        let sampleCount = min(learningTokens, vocabularySize)
        guard sampleCount > 1 else {
            return 1
        }

        let probabilities = exp(sortedLogprobs[0..., ..<sampleCount]).asArray(Float.self)
        var sumTIBI: Float = 0
        var sumTISquared: Float = 0
        for index in 0..<(sampleCount - 1) {
            let tI = log(Float(index + 2) / Float(index + 1))
            let current = max(probabilities[index], Self.epsilon)
            let next = max(probabilities[index + 1], Self.epsilon)
            let bI = log(current / next)
            sumTIBI += tI * bI
            sumTISquared += tI * tI
        }

        let sHat = max(sumTISquared > 0 ? sumTIBI / sumTISquared : 1, Self.epsilon)
        let epsilonHat = max(sHat - 1, Self.epsilon)
        let denominator = max(1 - pow(Float(vocabularySize), -epsilonHat), Self.epsilon)
        let k = pow((epsilonHat * pow(2, mu)) / denominator, 1 / sHat)

        guard k.isFinite else {
            return 1
        }
        return min(max(Int(k.rounded()), 1), vocabularySize)
    }

    private func updateState(sampledToken: MLXArray, logprobs: MLXArray) {
        eval(sampledToken)
        let tokenID = sampledToken.item(Int.self)
        let selectedIndex = MLXArray([Int32(tokenID)]).reshaped(1, 1)
        let selectedLogprob = takeAlong(logprobs, selectedIndex, axis: -1)
        eval(selectedLogprob)
        let observedSurprise = -selectedLogprob.item(Float.self) / Self.log2
        let error = observedSurprise - tau
        mu = max(Self.epsilon, mu - eta * error)
    }
}

private func makeRandomState(seed: Int?) -> MLXRandom.RandomState {
    guard let seed else { return MLXRandom.RandomState() }
    return MLXRandom.RandomState(seed: UInt64(bitPattern: Int64(seed)))
}

/// GPU-resident ring buffer of recent token IDs.
private struct TokenRing: @unchecked Sendable {
    private(set) var buffer: MLXArray
    private(set) var count = 0
    private var writeIndex = 0
    private let capacity: Int
    private let positions: MLXArray

    init(capacity: Int) {
        precondition(capacity > 0)
        self.capacity = capacity
        self.buffer = MLXArray.zeros([capacity], type: Int32.self)
        self.positions = MLXArray.arange(capacity)
    }

    var validTokens: MLXArray? {
        guard count > 0 else { return nil }
        return count < capacity ? buffer[..<count] : buffer
    }

    mutating func loadPrompt(_ prompt: MLXArray) {
        let promptTokenCount = prompt.dim(0)
        let promptTokens = prompt.asType(.int32)
        if promptTokenCount <= capacity {
            loadShortPrompt(promptTokens, promptTokenCount: promptTokenCount)
        } else {
            buffer = promptTokens[(-capacity)...].reshaped(-1)
            count = capacity
            writeIndex = 0
        }
    }

    mutating func append(_ token: MLXArray) {
        let mask = positions .== Int32(writeIndex)
        buffer = MLX.where(mask, token.asType(.int32), buffer)
        writeIndex = (writeIndex + 1) % capacity
        count = min(count + 1, capacity)
    }

    private mutating func loadShortPrompt(_ promptTokens: MLXArray, promptTokenCount: Int) {
        if promptTokenCount < capacity {
            let padding = MLXArray.zeros([capacity - promptTokenCount], type: Int32.self)
            buffer = concatenated([promptTokens.reshaped(-1), padding])
        } else {
            buffer = promptTokens.reshaped(-1)
        }
        count = promptTokenCount
        writeIndex = promptTokenCount % capacity
    }
}

/// Processor that implements a `repetitionPenalty`
internal struct RepetitionContext: LogitProcessor, @unchecked Sendable {
    private var ring: TokenRing
    /// penalty factor for repeating tokens
    let repetitionPenalty: Float

    internal init(repetitionPenalty: Float, repetitionContextSize: Int) {
        precondition(repetitionContextSize > 0)
        self.repetitionPenalty = repetitionPenalty
        self.ring = TokenRing(capacity: repetitionContextSize)
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else {
            return logits
        }

        var selectedLogits = logits[0..., indices]
        selectedLogits = MLX.where(
            selectedLogits .< 0,
            selectedLogits * repetitionPenalty,
            selectedLogits / repetitionPenalty
        )

        logits[0..., indices] = selectedLogits
        return logits
    }

    mutating internal func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies a flat additive penalty to recently seen tokens.
internal struct PresencePenaltyContext: LogitProcessor, @unchecked Sendable {
    private var ring: TokenRing
    let presencePenalty: Float

    internal init(presencePenalty: Float, presenceContextSize: Int) {
        precondition(presenceContextSize > 0)
        self.presencePenalty = presencePenalty
        self.ring = TokenRing(capacity: presenceContextSize)
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard let indices = ring.validTokens?.asType(.uint32) else {
            return logits
        }

        logits[0..., indices] = logits[0..., indices] - presencePenalty
        return logits
    }

    mutating internal func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies an additive penalty proportional to recent token frequency.
internal struct FrequencyPenaltyContext: LogitProcessor, @unchecked Sendable {
    private var ring: TokenRing
    let frequencyPenalty: Float

    internal init(frequencyPenalty: Float, frequencyContextSize: Int) {
        precondition(frequencyContextSize > 0)
        self.frequencyPenalty = frequencyPenalty
        self.ring = TokenRing(capacity: frequencyContextSize)
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        ring.loadPrompt(prompt)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard let validTokens = ring.validTokens else {
            return logits
        }

        let vocabularySize = logits.dim(-1)
        let ones = MLXArray.ones([validTokens.dim(0)], type: Float32.self)
        let histogram = MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[validTokens.asType(.int32)]
            .add(ones)

        return logits - (histogram * frequencyPenalty).reshaped(1, -1)
    }

    mutating internal func didSample(token: MLXArray) {
        ring.append(token)
    }
}

/// Processor that applies DRY sequence-repetition penalties at token level.
///
/// The suffix search mirrors llama.cpp's DRY sampler: restart sequences limit the active
/// context window, then a reverse Z-algorithm finds repeated suffixes in O(n).
internal struct DryPenaltyContext: LogitProcessor, @unchecked Sendable {
    private static let floatMaxLog: Float = 88.7228391

    private let multiplier: Float
    private let base: Float
    private let allowedLength: Int
    private let penaltyLastTokens: Int
    private let totalContextSize: Int?
    private let sequenceBreakersByHead: [Int: [[Int]]]
    private let singleTokenBreakerIds: Set<Int>
    private var lastTokens: [Int] = []

    internal init(
        configuration: DrySamplingConfiguration,
        totalContextSize: Int,
        sequenceBreakers: [[Int]]
    ) {
        self.multiplier = configuration.multiplier
        self.base = configuration.base
        self.allowedLength = max(0, Int(configuration.allowedLength))
        self.penaltyLastTokens = Int(configuration.penaltyLastTokens)
        self.totalContextSize = totalContextSize == Int.max ? nil : max(totalContextSize, 1)

        var breakersByHead: [Int: [[Int]]] = [:]
        var singleTokenBreakers: Set<Int> = []
        for sequence in sequenceBreakers where !sequence.isEmpty {
            let head = sequence[0]
            let tail = Array(sequence.dropFirst())
            breakersByHead[head, default: []].append(tail)
            if tail.isEmpty {
                singleTokenBreakers.insert(head)
            }
        }
        self.sequenceBreakersByHead = breakersByHead
        self.singleTokenBreakerIds = singleTokenBreakers
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        let tokens = prompt.reshaped(-1).asType(.int32).asArray(Int32.self).map(Int.init)
        lastTokens = suffixWithinTrackingLimit(tokens)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard multiplier != 0, base >= 1, penaltyLastTokens != 0 else {
            return logits
        }

        let effectiveLastTokens = effectivePenaltyLastTokens()
        let lastNRepeat = min(lastTokens.count, effectiveLastTokens)
        guard lastNRepeat > allowedLength else {
            return logits
        }

        let repetitionLimit = maximumRepetitionLimit(lastNRepeat: lastNRepeat)
        guard repetitionLimit >= allowedLength else {
            return logits
        }

        let repeatCounts = repeatedSuffixCounts(
            lastNRepeat: lastNRepeat,
            repetitionLimit: repetitionLimit
        )
        let penalties = dryPenaltyMap(
            repeatCounts: repeatCounts,
            lastNRepeat: lastNRepeat,
            vocabularySize: logits.dim(-1)
        )
        guard !penalties.isEmpty else {
            return logits
        }

        return logits + penaltyBias(from: penalties, vocabularySize: logits.dim(-1))
    }

    mutating internal func didSample(token: MLXArray) {
        eval(token)
        lastTokens.append(token.item(Int.self))
        trimToTrackingLimit()
    }

    private func effectivePenaltyLastTokens() -> Int {
        let contextLimit = totalContextSize ?? lastTokens.count
        if penaltyLastTokens == -1 {
            return contextLimit
        }
        return min(max(penaltyLastTokens, 0), contextLimit)
    }

    private func maximumRepetitionLimit(lastNRepeat: Int) -> Int {
        var repetitionLimit = lastNRepeat

        for indexFromEnd in 0 ..< lastNRepeat {
            let token = recentToken(indexFromEnd)
            guard let breakers = sequenceBreakersByHead[token] else {
                continue
            }

            var longestMatch = -1
            for tail in breakers {
                let tailLength = tail.count
                guard tailLength > longestMatch, tailLength <= indexFromEnd else {
                    continue
                }

                var matches = true
                for offset in 0 ..< tailLength
                    where tail[offset] != recentToken(indexFromEnd - offset - 1) {
                    matches = false
                    break
                }

                if matches {
                    longestMatch = tailLength
                }
            }

            if longestMatch >= 0 {
                repetitionLimit = indexFromEnd - longestMatch
                break
            }
        }

        return repetitionLimit
    }

    private func repeatedSuffixCounts(lastNRepeat: Int, repetitionLimit: Int) -> [Int] {
        let lastIndex = lastNRepeat - 1
        var repeatCounts = Array(repeating: 0, count: lastNRepeat)
        var right = 0
        var left = 0

        for offset in 1 ..< lastNRepeat {
            if offset > right {
                var count = 0
                while count + offset < lastNRepeat
                    && recentToken(count) == recentToken(count + offset) {
                    count += 1
                }
                repeatCounts[lastIndex - offset] = min(count, repetitionLimit)
                if count > 0 {
                    left = offset
                    right = offset + count - 1
                }
            } else {
                let pairIndex = offset - left
                let rightPartLength = right - offset + 1
                if repeatCounts[lastIndex - pairIndex] < rightPartLength {
                    repeatCounts[lastIndex - offset] = min(
                        repeatCounts[lastIndex - pairIndex],
                        repetitionLimit
                    )
                } else {
                    var index = right + 1
                    while index < lastNRepeat
                        && recentToken(index) == recentToken(index - offset) {
                        index += 1
                    }
                    repeatCounts[lastIndex - offset] = min(index - offset, repetitionLimit)
                    left = offset
                    right = index - 1
                }
            }
        }

        return repeatCounts
    }

    private func dryPenaltyMap(
        repeatCounts: [Int],
        lastNRepeat: Int,
        vocabularySize: Int
    ) -> [Int: Float] {
        var maxTokenRepeat: [Int: Int] = [:]
        for index in 0 ..< max(0, lastNRepeat - 1) {
            let repeatLength = repeatCounts[index]
            guard repeatLength >= allowedLength else {
                continue
            }

            let token = recentToken(lastNRepeat - 2 - index)
            maxTokenRepeat[token] = max(maxTokenRepeat[token] ?? 0, repeatLength)
        }

        guard !maxTokenRepeat.isEmpty else {
            return [:]
        }

        let maxExponent = base > 1.000001 ? Int(Self.floatMaxLog / log(base)) : 0
        var penalties: [Int: Float] = [:]
        for (token, repeatLength) in maxTokenRepeat
            where token >= 0 && token < vocabularySize && !singleTokenBreakerIds.contains(token) {
            var repeatExponent = repeatLength - allowedLength
            if maxExponent > 0 {
                repeatExponent = min(repeatExponent, maxExponent)
            }
            penalties[token] = -(multiplier * pow(base, Float(repeatExponent)))
        }
        return penalties
    }

    private func penaltyBias(from penalties: [Int: Float], vocabularySize: Int) -> MLXArray {
        let sortedPenalties = penalties.sorted { $0.key < $1.key }
        let indices = MLXArray(sortedPenalties.map(\.key)).asType(.int32)
        let values = MLXArray(sortedPenalties.map(\.value))

        return MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[indices]
            .add(values)
            .reshaped(1, -1)
    }

    private func recentToken(_ offsetFromEnd: Int) -> Int {
        lastTokens[lastTokens.count - offsetFromEnd - 1]
    }

    private func suffixWithinTrackingLimit(_ tokens: [Int]) -> [Int] {
        guard let trackingLimit = trackingLimit(), tokens.count > trackingLimit else {
            return tokens
        }
        return Array(tokens.suffix(trackingLimit))
    }

    private mutating func trimToTrackingLimit() {
        guard let trackingLimit = trackingLimit(), lastTokens.count > trackingLimit else {
            return
        }
        lastTokens.removeFirst(lastTokens.count - trackingLimit)
    }

    private func trackingLimit() -> Int? {
        if penaltyLastTokens == -1 {
            return totalContextSize
        }
        return max(penaltyLastTokens, 0)
    }
}

/// Processor that applies OpenAI-compatible additive logit bias by token id.
internal struct LogitBiasProcessor: LogitProcessor, @unchecked Sendable {
    private let indices: MLXArray
    private let values: MLXArray

    internal init(logitBias: [Int: Float]) {
        let sortedPairs = logitBias.sorted { $0.key < $1.key }
        self.indices = MLXArray(sortedPairs.map(\.key)).asType(.int32)
        self.values = MLXArray(sortedPairs.map(\.value))
    }

    mutating internal func prompt(_ prompt: MLXArray) {}

    mutating internal func process(logits: MLXArray) -> MLXArray {
        let vocabularySize = logits.dim(-1)
        let bias = MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[indices]
            .add(values)
        return logits + bias.reshaped(1, -1)
    }

    mutating internal func didSample(token: MLXArray) {}
}

/// Processor that hard-masks model-configured suppressed token ids.
internal struct SuppressTokensProcessor: LogitProcessor, @unchecked Sendable {
    private let indices: MLXArray
    private let negInf = MLXArray(-Float.infinity)

    internal init(tokenIds: Set<Int>) {
        self.indices = MLXArray(tokenIds.sorted()).asType(.uint32)
    }

    mutating internal func prompt(_ prompt: MLXArray) {}

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard indices.dim(0) > 0 else {
            return logits
        }
        let vocabularySize = logits.dim(-1)
        let bias = MLXArray.zeros([vocabularySize], type: Float32.self)
            .at[indices]
            .add(negInf)
        return logits + bias.reshaped(1, -1)
    }

    mutating internal func didSample(token: MLXArray) {}
}

internal enum ReasoningBudgetSampleResult: Sendable, Equatable {
    case counted(reasoningTokenCount: Int)
    case budgetReached(reasoningTokenCount: Int, nextTokenID: Int)
    case forcing(nextTokenID: Int)
    case closed(forced: Bool)
}

internal struct ReasoningBudgetState: Sendable, Equatable {
    private let maximumReasoningTokens: Int
    private let endTokenIds: [Int]
    private var recentTokenIds: [Int] = []
    private var forceIndex: Int?

    private(set) var reasoningTokenCount = 0
    private(set) var isClosed = false

    internal init(maximumReasoningTokens: Int, endTokenIds: [Int]) {
        self.maximumReasoningTokens = max(1, maximumReasoningTokens)
        self.endTokenIds = endTokenIds.filter { $0 >= 0 }
    }

    internal var nextForcedTokenID: Int? {
        guard !isClosed, !endTokenIds.isEmpty else {
            return nil
        }
        if let forceIndex {
            return endTokenIds[forceIndex]
        }
        return reasoningTokenCount >= maximumReasoningTokens ? endTokenIds[0] : nil
    }

    internal mutating func didSample(_ tokenID: Int) -> ReasoningBudgetSampleResult {
        guard !isClosed else {
            return .closed(forced: false)
        }
        if let expectedTokenID = nextForcedTokenID {
            return didSampleForced(tokenID, expectedTokenID: expectedTokenID)
        }

        reasoningTokenCount += 1
        remember(tokenID)
        if recentTokenIds == endTokenIds {
            isClosed = true
            return .closed(forced: false)
        }
        if reasoningTokenCount >= maximumReasoningTokens, let nextForcedTokenID {
            return .budgetReached(
                reasoningTokenCount: reasoningTokenCount,
                nextTokenID: nextForcedTokenID
            )
        }
        return .counted(reasoningTokenCount: reasoningTokenCount)
    }

    private mutating func didSampleForced(
        _ tokenID: Int,
        expectedTokenID: Int
    ) -> ReasoningBudgetSampleResult {
        let currentForceIndex = forceIndex ?? 0
        guard tokenID == expectedTokenID else {
            forceIndex = currentForceIndex
            return .forcing(nextTokenID: expectedTokenID)
        }

        let nextForceIndex = currentForceIndex + 1
        if nextForceIndex >= endTokenIds.count {
            forceIndex = nil
            isClosed = true
            return .closed(forced: true)
        }
        forceIndex = nextForceIndex
        return .forcing(nextTokenID: endTokenIds[nextForceIndex])
    }

    private mutating func remember(_ tokenID: Int) {
        guard !endTokenIds.isEmpty else {
            return
        }
        recentTokenIds.append(tokenID)
        if recentTokenIds.count > endTokenIds.count {
            recentTokenIds.removeFirst(recentTokenIds.count - endTokenIds.count)
        }
    }
}

/// Forces the model's reasoning-close marker once a token budget is reached.
internal struct ReasoningBudgetProcessor: LogitProcessor, @unchecked Sendable {
    private var state: ReasoningBudgetState
    private let negInf = MLXArray(-Float.infinity)

    internal init(maximumReasoningTokens: Int, endTokenIds: [Int]) {
        self.state = ReasoningBudgetState(
            maximumReasoningTokens: maximumReasoningTokens,
            endTokenIds: endTokenIds
        )
    }

    mutating internal func prompt(_ prompt: MLXArray) {}

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard let tokenID = state.nextForcedTokenID else {
            return logits
        }
        guard tokenID < logits.dim(-1) else {
            MLXGenerationDiagnostics.recordReasoningBudget(.init(
                stage: .invalidEndToken,
                reasoningTokenCount: state.reasoningTokenCount,
                forcedTokenID: tokenID,
                message: "Reasoning end token is outside the logits vocabulary"
            ))
            return logits
        }

        MLXGenerationDiagnostics.recordReasoningBudget(.init(
            stage: .maskApplied,
            reasoningTokenCount: state.reasoningTokenCount,
            forcedTokenID: tokenID,
            message: nil
        ))
        let indices = MLXArray([tokenID]).asType(.int32)
        let masked = MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)
        masked[0..., indices] = logits[0..., indices]
        return masked
    }

    mutating internal func didSample(token: MLXArray) {
        let tokenID = token.item(Int.self)
        let result = state.didSample(tokenID)
        guard let snapshot = snapshot(for: result, tokenID: tokenID) else {
            return
        }
        MLXGenerationDiagnostics.recordReasoningBudget(snapshot)
    }

    private func snapshot(
        for result: ReasoningBudgetSampleResult,
        tokenID: Int
    ) -> MLXReasoningBudgetSnapshot? {
        switch result {
        case .counted(let reasoningTokenCount):
            return .init(
                stage: .tokenCounted,
                reasoningTokenCount: reasoningTokenCount,
                forcedTokenID: nil,
                message: nil
            )

        case .budgetReached(let reasoningTokenCount, let nextTokenID):
            return .init(
                stage: .budgetReached,
                reasoningTokenCount: reasoningTokenCount,
                forcedTokenID: nextTokenID,
                message: nil
            )

        case .forcing(let nextTokenID):
            return .init(
                stage: .forcingEndMarker,
                reasoningTokenCount: state.reasoningTokenCount,
                forcedTokenID: nextTokenID,
                message: nil
            )

        case .closed(let forced):
            return .init(
                stage: forced ? .forcedClosed : .naturallyClosed,
                reasoningTokenCount: state.reasoningTokenCount,
                forcedTokenID: tokenID,
                message: nil
            )
        }
    }
}

/// Composes active logit processors in a deterministic order.
internal struct PenaltyProcessor: LogitProcessor, @unchecked Sendable {
    var logitBiasContext: LogitBiasProcessor?
    var suppressTokenContext: SuppressTokensProcessor?
    var reasoningBudgetContext: ReasoningBudgetProcessor?
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?
    var dryContext: DryPenaltyContext?
    var grammarContext: GrammarConstrainedLogitProcessor?

    internal init(
        logitBiasContext: LogitBiasProcessor?,
        suppressTokenContext: SuppressTokensProcessor?,
        reasoningBudgetContext: ReasoningBudgetProcessor?,
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?,
        dryContext: DryPenaltyContext?,
        grammarContext: GrammarConstrainedLogitProcessor?
    ) {
        self.logitBiasContext = logitBiasContext
        self.suppressTokenContext = suppressTokenContext
        self.reasoningBudgetContext = reasoningBudgetContext
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
        self.dryContext = dryContext
        self.grammarContext = grammarContext
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        logitBiasContext?.prompt(prompt)
        suppressTokenContext?.prompt(prompt)
        reasoningBudgetContext?.prompt(prompt)
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
        dryContext?.prompt(prompt)
        grammarContext?.prompt(prompt)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = logitBiasContext?.process(logits: logits) ?? logits
        logits = suppressTokenContext?.process(logits: logits) ?? logits
        logits = reasoningBudgetContext?.process(logits: logits) ?? logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        logits = dryContext?.process(logits: logits) ?? logits
        logits = grammarContext?.process(logits: logits) ?? logits

        return logits
    }

    mutating internal func didSample(token: MLXArray) {
        logitBiasContext?.didSample(token: token)
        suppressTokenContext?.didSample(token: token)
        reasoningBudgetContext?.didSample(token: token)
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
        dryContext?.didSample(token: token)
        grammarContext?.didSample(token: token)
    }
}

/// Pull-based token generator used by the synchronous and streaming generation entry points.
internal struct TokenIterator: Sequence, IteratorProtocol {
    let model: any LanguageModel
    var state: LMOutput.State?

    var y: LMInput.Text
    var cache: [KVCache]
    var processor: LogitProcessor?
    let sampler: LogitSampler

    var tokenCount = 0
    let maxTokens: Int?

    // Cache quantization parameters
    let kvBits: Int?
    let kvGroupSize: Int
    let quantizedKVStart: Int
    let quantizedKVSkipLastLayer: Bool

    /// Initialize a `TokenIterator` with the given tokens. Note: this has been
    /// replaced with ``init(input:model:cache:parameters:)``.
    ///
    /// - Parameters:
    ///   - prompt: the prompt tokens
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    @available(*, deprecated, message: "please use init(input:model:cache:parameters:)")
    internal init(
        prompt: MLXArray, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters
    ) throws {
        self.model = model
        self.y = .init(tokens: prompt)
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = try parameters.processor()
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.quantizedKVSkipLastLayer = parameters.quantizedKVSkipLastLayer

        try prepare(input: .init(text: y), windowSize: parameters.prefillStepSize)
    }

    /// Initialize a `TokenIterator` with the given input.
    ///
    /// If more control is needed over the generation,
    /// ``init(input:model:cache:processor:sampler:prefillStepSize:)``
    /// allows a caller to specify ``LogitProcessor`` and ``LogitSampler``
    /// directly.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - parameters: the generation parameters
    internal init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        processorPrompt: LMInput.Text? = nil,
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: parameters)

        self.processor = try parameters.processor(grammarCompiler: grammarCompiler)
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens

        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.quantizedKVSkipLastLayer = parameters.quantizedKVSkipLastLayer

        try prepare(
            input: input,
            processorPrompt: processorPrompt,
            windowSize: parameters.prefillStepSize
        )
    }

    /// Initialize a `TokenIterator` with the given input and logit handling.
    ///
    /// - Parameters:
    ///   - input: language model input
    ///   - model: the ``LanguageModel``
    ///   - cache: optional ``KVCache``
    ///   - processor: the logit processor
    ///   - sampler: the logit sampler
    ///   - prefillStepSize: optional prefill step size
    ///   - maxTokens: maximum number of tokens to generate
    internal init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        processor: LogitProcessor?, sampler: LogitSampler,
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize,
        maxTokens: Int? = nil
    ) throws {
        self.model = model
        self.y = input.text
        self.cache = cache ?? model.newCache(parameters: nil)

        self.processor = processor
        self.sampler = sampler
        self.maxTokens = maxTokens

        // No cache quantization for this direct initialization
        self.kvBits = nil
        self.kvGroupSize = GenerationConstants.defaultKVCacheGroupSize
        self.quantizedKVStart = 0
        self.quantizedKVSkipLastLayer = false

        try prepare(input: input, windowSize: prefillStepSize)
    }

    mutating func prepare(
        input: LMInput,
        processorPrompt: LMInput.Text? = nil,
        windowSize: Int? = nil
    ) throws {
        processor?.prompt((processorPrompt ?? input.text).tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

            // evaluate the remainder of the prompt -- this primes the pump
            let token = step(previous: y)
            y = .init(tokens: token)
            asyncEval(y.tokens)

        case .logits(let result):
            y = .init(tokens: convertToToken(logits: result.logits))
            asyncEval(y.tokens)

            break
        }
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "prepared", cache: cache)
    }

    mutating func convertToToken(logits: MLXArray) -> MLXArray {
        // process the logits (one hot array of possible tokens)
        var logits = logits[0..., -1, 0...]
        logits = processor?.process(logits: logits) ?? logits

        // transform logits back to a token
        let y = sampler.sample(logits: logits)

        processor?.didSample(token: y)

        return y
    }

    /// Evaluate the next token and return the new token (y), updating cache state
    mutating func step(previous: LMInput.Text) -> MLXArray {
        let token: MLXArray
        if processor == nil,
            sampler is ArgMaxSampler,
            let greedyModel = model as? any GreedyTokenModel {
            let result = greedyModel.greedyToken(
                previous,
                cache: cache.isEmpty ? nil : cache,
                state: state
            )
            self.state = result.state
            token = result.token
        } else {
            let result = model(
                previous[text: .newAxis],
                cache: cache.isEmpty ? nil : cache,
                state: state
            )
            self.state = result.state
            token = convertToToken(logits: result.logits)
        }

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            skipLastLayer: quantizedKVSkipLastLayer
        )
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "step", cache: cache)

        return token
    }

    mutating internal func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }

        // save current value -- this will be returned
        let previousY = y

        // compute the next state and async eval the next token
        let token = step(previous: previousY)
        y = .init(tokens: token)
        asyncEval(token)

        tokenCount += 1

        return previousY.tokens.item(Int.self)
    }
}

/// Token iterator that uses a draft model to propose tokens and the main model to verify them in batches.
internal struct SpeculativeTokenIterator: Sequence, IteratorProtocol {
    private var y: LMInput.Text
    private var draftY: LMInput.Text

    private let mainModel: any LanguageModel
    private let draftModel: any LanguageModel

    private var mainState: LMOutput.State?
    private var mainCache: [KVCache]
    private var draftCache: [KVCache]
    private let quantizeKVCache: (inout [KVCache]) -> Void

    private var processor: LogitProcessor?
    private let sampler: LogitSampler

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0

    internal private(set) var tokenCount = 0
    internal let maxTokens: Int?
    internal let numDraftTokens: Int
    private var adaptiveDraftTokens: Int
    internal var cacheForPromptReuse: PromptCacheReusableState {
        PromptCacheReusableState(cache: mainCache, draftCache: draftCache)
    }

    internal init(
        input: LMInput,
        mainModel: any LanguageModel,
        draftModel: any LanguageModel,
        mainCache: [KVCache]? = nil,
        draftCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processorPrompt: LMInput.Text? = nil,
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws {
        self.y = input.text
        self.draftY = input.text
        self.mainModel = mainModel
        self.draftModel = draftModel
        self.mainCache = mainCache ?? mainModel.newCache(parameters: parameters)
        self.draftCache = draftCache ?? draftModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.mainCache), canTrimPromptCache(self.draftCache) else {
            throw KVCacheError(message: "Speculative decoding requires trimmable KV caches.")
        }

        self.processor = try parameters.processor(grammarCompiler: grammarCompiler)
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = Swift.max(0, numDraftTokens)
        self.adaptiveDraftTokens = Swift.max(0, numDraftTokens)
        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                skipLastLayer: parameters.quantizedKVSkipLastLayer
            )
        }

        try prepare(
            input: input,
            processorPrompt: processorPrompt,
            windowSize: parameters.prefillStepSize
        )
    }

    internal mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if let pendingToken = nextPendingToken() {
            return pendingToken
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        speculateRound()

        return nextPendingToken()
    }

    private mutating func prepare(
        input: LMInput,
        processorPrompt: LMInput.Text?,
        windowSize: Int?
    ) throws {
        processor?.prompt((processorPrompt ?? input.text).tokens)
        try prepareMainModel(input: input, windowSize: windowSize)
        try prepareDraftModel(input: input, windowSize: windowSize)
    }

    private mutating func prepareMainModel(input: LMInput, windowSize: Int?) throws {
        switch try mainModel.prepare(input, cache: mainCache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

        case .logits(let result):
            let token = sampleNextToken(logits: result.logits, processor: &processor)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            mainState = result.state
            asyncEval(y.tokens)
        }
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "speculativePreparedMain", cache: mainCache)
    }

    private mutating func prepareDraftModel(input: LMInput, windowSize: Int?) throws {
        switch try draftModel.prepare(input, cache: draftCache, windowSize: windowSize) {
        case .tokens(let tokens):
            draftY = tokens

        case .logits(let result):
            var draftProcessor = processor
            let token = sampleNextToken(logits: result.logits, processor: &draftProcessor)
            draftY = .init(tokens: token)
            asyncEval(draftY.tokens)
        }
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "speculativePreparedDraft", cache: draftCache)
    }

    private mutating func speculateRound() {
        let remainingTokens = maxTokens.map { $0 - tokenCount } ?? adaptiveDraftTokens
        let draftCount = Swift.min(remainingTokens, adaptiveDraftTokens)
        guard draftCount > 0 else {
            return
        }

        let draftTokens = generateDraftTokens(count: draftCount)
        let mainTokens = verifyDraftTokens(draftTokens)
        acceptVerifiedTokens(mainTokens: mainTokens, draftTokens: draftTokens)
    }

    private mutating func generateDraftTokens(count: Int) -> [MLXArray] {
        var draftProcessor = processor
        var draftTokens: [MLXArray] = []
        draftTokens.reserveCapacity(count)

        for _ in 0 ..< count {
            let result = draftModel(draftY[text: .newAxis], cache: draftCache, state: nil)
            let token = sampleNextToken(logits: result.logits, processor: &draftProcessor)
            draftProcessor?.didSample(token: token)
            asyncEval(token)
            draftTokens.append(token)
            draftY = .init(tokens: token)
        }

        MLXGenerationDiagnostics.recordCacheSnapshot(label: "speculativeDraftStep", cache: draftCache)
        return draftTokens
    }

    private mutating func verifyDraftTokens(_ draftTokens: [MLXArray]) -> MLXArray {
        let verifyTokens = [y.tokens] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let verifyStart = verifyInput.tokens.dim(0) - (draftTokens.count + 1)
        let result = mainModel(
            verifyInput[text: .newAxis],
            cache: mainCache,
            state: mainState
        )
        mainState = result.state
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "speculativeMainStep", cache: mainCache)

        guard var verifyProcessor = processor else {
            let logits = result.logits[0..., verifyStart..., 0...].squeezed(axis: 0)
            return sampler.sample(logits: logits)
        }

        let sampled = (0 ..< draftTokens.count + 1).map { index in
            var logits = result.logits[0..., verifyStart + index, 0...]
            logits = verifyProcessor.process(logits: logits)
            let token = sampler.sample(logits: logits)
            verifyProcessor.didSample(token: token)
            return token
        }
        return concatenated(sampled)
    }

    private mutating func acceptVerifiedTokens(mainTokens: MLXArray, draftTokens: [MLXArray]) {
        eval(mainTokens, draftTokens)
        let mainTokenIDs = mainTokens.asArray(Int.self)
        let draftTokenIDs = concatenated(draftTokens).asArray(Int.self)
        var acceptedCount = 0

        for index in 0 ..< draftTokens.count {
            guard mainTokenIDs[index] == draftTokenIDs[index] else {
                break
            }

            processor?.didSample(token: draftTokens[index])
            pendingTokens.append(mainTokenIDs[index])
            acceptedCount += 1
        }

        let finalToken = mainTokens[acceptedCount ... acceptedCount]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokenIDs[acceptedCount])

        let rejectedCount = draftTokens.count - acceptedCount
        MLXGenerationDiagnostics.recordSpeculativeDecoding(
            numDraftTokens: draftTokens.count,
            acceptedDraftTokens: acceptedCount,
            rejectedDraftTokens: rejectedCount,
            emittedTokens: acceptedCount + 1
        )
        updateAdaptiveDraftWindow(acceptedCount: acceptedCount, attemptedCount: draftTokens.count)
        trimPromptCache(mainCache, numTokens: rejectedCount)
        trimPromptCache(draftCache, numTokens: Swift.max(rejectedCount - 1, 0))
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)
        if acceptedCount == draftTokens.count, let lastDraftToken = draftTokens.last {
            draftY = .init(tokens: concatenated([lastDraftToken.reshaped([1]), finalToken]))
        }
    }

    private mutating func updateAdaptiveDraftWindow(acceptedCount: Int, attemptedCount: Int) {
        guard numDraftTokens > 1, attemptedCount > 0 else {
            return
        }
        if acceptedCount == attemptedCount {
            adaptiveDraftTokens = Swift.min(numDraftTokens, adaptiveDraftTokens + 1)
            return
        }
        if acceptedCount * 2 < attemptedCount {
            adaptiveDraftTokens = Swift.max(1, adaptiveDraftTokens - 1)
        }
    }

    private mutating func nextPendingToken() -> Int? {
        guard pendingIndex < pendingTokens.count else {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    private func sampleNextToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        let logits = logits[0..., -1, 0...]
        return sampleToken(logits: logits, processor: &processor)
    }

    private func sampleToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        return sampler.sample(logits: logits)
    }
}

internal struct NativeMTPTokenIterator: Sequence, IteratorProtocol {
    private var y: LMInput.Text
    private let model: any NativeMTPModel
    private var state: LMOutput.State?
    private var cache: [KVCache]
    private let quantizeKVCache: (inout [KVCache]) -> Void
    private let parameters: GenerateParameters

    private var processor: LogitProcessor?
    private let sampler: LogitSampler

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0

    internal private(set) var tokenCount = 0
    internal let maxTokens: Int?
    internal let numDraftTokens: Int
    internal var cacheForPromptReuse: PromptCacheReusableState {
        PromptCacheReusableState(cache: cache)
    }

    internal init(
        input: LMInput,
        model: any NativeMTPModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processorPrompt: LMInput.Text? = nil,
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws {
        self.y = input.text
        self.model = model
        self.cache = cache ?? model.newCache(parameters: parameters)
        self.parameters = parameters
        guard canTrimPromptCache(self.cache) else {
            throw KVCacheError(message: "Native MTP decoding requires trimmable KV caches.")
        }

        self.processor = try parameters.processor(grammarCompiler: grammarCompiler)
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = Swift.max(0, numDraftTokens)
        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                skipLastLayer: parameters.quantizedKVSkipLastLayer
            )
        }

        try prepare(
            input: input,
            processorPrompt: processorPrompt,
            windowSize: parameters.prefillStepSize
        )
    }

    internal mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if let pendingToken = nextPendingToken() {
            return pendingToken
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        nativeMTPRound()

        return nextPendingToken()
    }

    private mutating func prepare(
        input: LMInput,
        processorPrompt: LMInput.Text?,
        windowSize: Int?
    ) throws {
        processor?.prompt((processorPrompt ?? input.text).tokens)

        switch try model.prepare(input, cache: cache, windowSize: windowSize) {
        case .tokens(let tokens):
            y = tokens

        case .logits(let result):
            let token = sampleNextToken(logits: result.logits, processor: &processor)
            processor?.didSample(token: token)
            y = .init(tokens: token)
            state = result.state
            asyncEval(y.tokens)
        }
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "nativeMTPPrepared", cache: cache)
    }

    private mutating func nativeMTPRound() {
        let remainingTokens = maxTokens.map { $0 - tokenCount } ?? (numDraftTokens + 1)
        guard remainingTokens > 0 else {
            return
        }

        let mainOutput = model.nativeMTPMainOutput(
            y[text: .newAxis],
            cache: cache.isEmpty ? nil : cache,
            state: state
        )
        state = mainOutput.state
        let firstToken = sampleNextToken(logits: mainOutput.logits, processor: &processor)

        let draftCount = Swift.min(Swift.max(remainingTokens - 1, 0), numDraftTokens)
        guard draftCount > 0 else {
            acceptMainOnly(firstToken)
            return
        }

        let draftTokens = generateDraftTokens(
            count: draftCount,
            firstToken: firstToken,
            mainHiddenStates: mainOutput.hiddenStates
        )
        guard !draftTokens.isEmpty else {
            acceptMainOnly(firstToken)
            return
        }

        let mainTokens = verifyDraftTokens(firstToken: firstToken, draftTokens: draftTokens)
        acceptVerifiedTokens(firstToken: firstToken, mainTokens: mainTokens, draftTokens: draftTokens)
    }

    private mutating func generateDraftTokens(
        count: Int,
        firstToken: MLXArray,
        mainHiddenStates: MLXArray
    ) -> [MLXArray] {
        var draftProcessor = processor
        draftProcessor?.didSample(token: firstToken)

        var draftTokens: [MLXArray] = []
        draftTokens.reserveCapacity(count)
        let lastHiddenStateIndex = mainHiddenStates.dim(1) - 1
        var hiddenStates = mainHiddenStates[0..., lastHiddenStateIndex ..< lastHiddenStateIndex + 1, 0...]
        var nextToken = firstToken
        let mtpCache = model.makeMTPCache(parameters: parameters)

        for _ in 0 ..< count {
            guard let draftOutput = model.nativeMTPDraftOutput(
                hiddenStates: hiddenStates,
                nextTokenIDs: nextToken.reshaped([1, 1]),
                cache: mtpCache.isEmpty ? nil : mtpCache
            ) else {
                break
            }
            let token = sampleNextToken(logits: draftOutput.logits, processor: &draftProcessor)
            draftProcessor?.didSample(token: token)
            asyncEval(token)
            draftTokens.append(token)
            nextToken = token
            hiddenStates = draftOutput.hiddenStates
        }

        MLXGenerationDiagnostics.recordCacheSnapshot(label: "nativeMTPDraftStep", cache: mtpCache)
        return draftTokens
    }

    private mutating func verifyDraftTokens(
        firstToken: MLXArray,
        draftTokens: [MLXArray]
    ) -> MLXArray {
        let verifyTokens = [firstToken] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let result = model(
            verifyInput[text: .newAxis],
            cache: cache.isEmpty ? nil : cache,
            state: state
        )
        state = result.state
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "nativeMTPMainStep", cache: cache)

        guard var verifyProcessor = processor else {
            let logits = result.logits[0..., 0..., 0...].squeezed(axis: 0)
            return sampler.sample(logits: logits)
        }

        verifyProcessor.didSample(token: firstToken)
        let sampled = (0 ..< draftTokens.count + 1).map { index in
            var logits = result.logits[0..., index, 0...]
            logits = verifyProcessor.process(logits: logits)
            let token = sampler.sample(logits: logits)
            verifyProcessor.didSample(token: token)
            return token
        }
        return concatenated(sampled)
    }

    private mutating func acceptVerifiedTokens(
        firstToken: MLXArray,
        mainTokens: MLXArray,
        draftTokens: [MLXArray]
    ) {
        eval(firstToken, mainTokens, draftTokens)
        let firstTokenID = firstToken.item(Int.self)
        let mainTokenIDs = mainTokens.asArray(Int.self)
        let draftTokenIDs = concatenated(draftTokens).asArray(Int.self)
        var acceptedCount = 0

        processor?.didSample(token: firstToken)
        pendingTokens.append(firstTokenID)

        for index in 0 ..< draftTokens.count {
            guard mainTokenIDs[index] == draftTokenIDs[index] else {
                break
            }

            processor?.didSample(token: draftTokens[index])
            pendingTokens.append(mainTokenIDs[index])
            acceptedCount += 1
        }

        let finalToken = mainTokens[acceptedCount ... acceptedCount]
        processor?.didSample(token: finalToken)
        pendingTokens.append(mainTokenIDs[acceptedCount])

        trimPromptCache(cache, numTokens: draftTokens.count - acceptedCount)
        quantizeKVCache(&cache)
        y = .init(tokens: finalToken)
    }

    private mutating func acceptMainOnly(_ firstToken: MLXArray) {
        eval(firstToken)
        processor?.didSample(token: firstToken)
        pendingTokens.append(firstToken.item(Int.self))
        quantizeKVCache(&cache)
        y = .init(tokens: firstToken)
    }

    private mutating func nextPendingToken() -> Int? {
        guard pendingIndex < pendingTokens.count else {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    private func sampleNextToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        let logits = logits[0..., -1, 0...]
        return sampleToken(logits: logits, processor: &processor)
    }

    private func sampleToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        return sampler.sample(logits: logits)
    }
}

internal struct SharedKVMTPTokenIterator: Sequence, IteratorProtocol {
    private var lastBonusToken: MLXArray?
    private var draftHiddenStates: MLXArray?
    private var sharedKVStates: [String: SharedKVState] = [:]
    private var kvOffset = 0

    private let targetModel: any SharedKVSpeculativeTargetModel
    private let draftModel: any SharedKVSpeculativeDraftModel
    private var targetState: LMOutput.State?
    private var targetCache: [KVCache]
    private let quantizeKVCache: (inout [KVCache]) -> Void

    private var processor: LogitProcessor?
    private let sampler: LogitSampler

    private var pendingTokens: [Int] = []
    private var pendingIndex = 0

    internal private(set) var tokenCount = 0
    internal let maxTokens: Int?
    internal let numDraftTokens: Int
    private var adaptiveDraftTokens: Int
    internal var cacheForPromptReuse: PromptCacheReusableState {
        PromptCacheReusableState(cache: targetCache)
    }

    internal init(
        input: LMInput,
        targetModel: any SharedKVSpeculativeTargetModel,
        draftModel: any SharedKVSpeculativeDraftModel,
        targetCache: [KVCache]? = nil,
        parameters: GenerateParameters,
        numDraftTokens: Int,
        processorPrompt: LMInput.Text? = nil,
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws {
        self.targetModel = targetModel
        self.draftModel = draftModel
        self.targetCache = targetCache ?? targetModel.newCache(parameters: parameters)
        guard canTrimPromptCache(self.targetCache) else {
            throw KVCacheError(message: "Shared-KV MTP decoding requires trimmable KV caches.")
        }

        self.processor = try parameters.processor(grammarCompiler: grammarCompiler)
        self.sampler = parameters.sampler()
        self.maxTokens = parameters.maxTokens
        self.numDraftTokens = Swift.max(0, numDraftTokens)
        self.adaptiveDraftTokens = Swift.max(0, numDraftTokens)
        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart,
                skipLastLayer: parameters.quantizedKVSkipLastLayer
            )
        }

        try prepare(
            input: input,
            processorPrompt: processorPrompt,
            windowSize: parameters.prefillStepSize
        )
    }

    internal mutating func next() -> Int? {
        if let maxTokens, tokenCount >= maxTokens {
            return nil
        }
        if let pendingToken = nextPendingToken() {
            return pendingToken
        }

        pendingTokens.removeAll(keepingCapacity: true)
        pendingIndex = 0
        sharedKVMTPRound()

        return nextPendingToken()
    }

    private mutating func prepare(
        input: LMInput,
        processorPrompt: LMInput.Text?,
        windowSize: Int?
    ) throws {
        processor?.prompt((processorPrompt ?? input.text).tokens)
        let output = try targetModel.speculativePrepare(
            input,
            cache: targetCache,
            windowSize: windowSize
        )
        targetState = output.state
        sharedKVStates = output.sharedKVStates
        kvOffset = cacheOffset(targetCache)

        let firstBonus = sampleNextToken(logits: output.logits, processor: &processor)
        processor?.didSample(token: firstBonus)
        lastBonusToken = firstBonus
        draftHiddenStates = lastDraftHidden(from: output.hiddenStates)
        pendingTokens.append(firstBonus.item(Int.self))
        asyncEval(firstBonus)
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "sharedKVMTPPrepared", cache: targetCache)
    }

    private mutating func sharedKVMTPRound() {
        guard let lastBonusToken, let draftHiddenStates else {
            return
        }

        let remainingTokens = maxTokens.map { $0 - tokenCount } ?? (adaptiveDraftTokens + 1)
        guard remainingTokens > 0 else {
            return
        }

        let draftCount = Swift.min(Swift.max(remainingTokens - 1, 0), adaptiveDraftTokens)
        guard draftCount > 0 else {
            acceptMainOnly(lastBonusToken: lastBonusToken)
            return
        }

        let draftTokens = generateDraftTokens(
            count: draftCount,
            lastBonusToken: lastBonusToken,
            hiddenStates: draftHiddenStates
        )
        guard !draftTokens.isEmpty else {
            acceptMainOnly(lastBonusToken: lastBonusToken)
            return
        }

        let verification = verifyDraftTokens(lastBonusToken: lastBonusToken, draftTokens: draftTokens)
        acceptVerifiedTokens(verification: verification, draftTokens: draftTokens)
    }

    private mutating func generateDraftTokens(
        count: Int,
        lastBonusToken: MLXArray,
        hiddenStates: MLXArray
    ) -> [MLXArray] {
        var draftProcessor = processor
        var draftTokens: [MLXArray] = []
        draftTokens.reserveCapacity(count)

        var nextToken = lastBonusToken
        var hiddenStates = hiddenStates
        let position = Swift.max(kvOffset - 1, 0)
        let useGreedyDraft = processor == nil && sampler is ArgMaxSampler

        for _ in 0 ..< count {
            let tokenEmbeddings = targetModel.speculativeTokenEmbeddings(
                nextToken.reshaped([1, 1])
            )
            let token: MLXArray
            if useGreedyDraft,
                let output = draftModel.sharedKVGreedyDraftOutput(
                    tokenEmbeddings: tokenEmbeddings,
                    hiddenStates: hiddenStates,
                    sharedKVStates: sharedKVStates,
                    position: position
                ) {
                token = output.token
                hiddenStates = output.hiddenStates
            } else {
                let output = draftModel.sharedKVDraftOutput(
                    tokenEmbeddings: tokenEmbeddings,
                    hiddenStates: hiddenStates,
                    sharedKVStates: sharedKVStates,
                    position: position
                )
                token = sampleNextToken(logits: output.logits, processor: &draftProcessor)
                draftProcessor?.didSample(token: token)
                hiddenStates = output.hiddenStates
            }
            let flatToken = token.reshaped([-1])
            asyncEval(flatToken)
            draftTokens.append(flatToken)
            nextToken = flatToken
        }

        MLXGenerationDiagnostics.recordCacheSnapshot(label: "sharedKVMTPDraftStep", cache: [])
        return draftTokens
    }

    private mutating func verifyDraftTokens(
        lastBonusToken: MLXArray,
        draftTokens: [MLXArray]
    ) -> (targetTokens: MLXArray, output: SharedKVTargetOutput) {
        let verifyTokens = [lastBonusToken] + draftTokens
        let verifyInput = LMInput.Text(tokens: concatenated(verifyTokens))
        let output = targetModel.speculativeTargetOutput(
            verifyInput,
            cache: targetCache.isEmpty ? nil : targetCache,
            state: targetState
        )
        targetState = output.state
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "sharedKVMTPVerifyStep", cache: targetCache)

        guard var verifyProcessor = processor else {
            let logits = output.logits[0..., 0..., 0...].squeezed(axis: 0)
            return (sampler.sample(logits: logits), output)
        }

        let sampled = (0 ..< draftTokens.count + 1).map { index in
            var logits = output.logits[0..., index, 0...]
            logits = verifyProcessor.process(logits: logits)
            let token = sampler.sample(logits: logits)
            verifyProcessor.didSample(token: token)
            return token
        }
        return (concatenated(sampled), output)
    }

    private mutating func acceptVerifiedTokens(
        verification: (targetTokens: MLXArray, output: SharedKVTargetOutput),
        draftTokens: [MLXArray]
    ) {
        eval(verification.targetTokens, draftTokens)
        let targetTokenIDs = verification.targetTokens.asArray(Int.self)
        let draftTokenIDs = concatenated(draftTokens).asArray(Int.self)
        var acceptedCount = 0

        for index in 0 ..< draftTokens.count {
            guard targetTokenIDs[index] == draftTokenIDs[index] else {
                break
            }

            processor?.didSample(token: draftTokens[index])
            pendingTokens.append(targetTokenIDs[index])
            acceptedCount += 1
        }

        let finalToken = verification.targetTokens[acceptedCount ... acceptedCount]
        processor?.didSample(token: finalToken)
        pendingTokens.append(targetTokenIDs[acceptedCount])

        let rejectedCount = draftTokens.count - acceptedCount
        MLXGenerationDiagnostics.recordSpeculativeDecoding(
            numDraftTokens: draftTokens.count,
            acceptedDraftTokens: acceptedCount,
            rejectedDraftTokens: rejectedCount,
            emittedTokens: acceptedCount + 1
        )
        updateAdaptiveDraftWindow(acceptedCount: acceptedCount, attemptedCount: draftTokens.count)
        trimPromptCache(targetCache, numTokens: rejectedCount)
        quantizeKVCache(&targetCache)
        kvOffset += acceptedCount + 1
        sharedKVStates = trimmedSharedKVStates(
            verification.output.sharedKVStates,
            rejectedCount: rejectedCount
        )
        draftHiddenStates = lastDraftHidden(
            from: verification.output.hiddenStates[0..., acceptedCount ..< acceptedCount + 1, 0...]
        )
        lastBonusToken = finalToken
    }

    private mutating func updateAdaptiveDraftWindow(acceptedCount: Int, attemptedCount: Int) {
        guard numDraftTokens > 1, attemptedCount > 0 else {
            return
        }
        if acceptedCount == attemptedCount {
            adaptiveDraftTokens = Swift.min(numDraftTokens, adaptiveDraftTokens + 1)
            return
        }
        if acceptedCount * 2 < attemptedCount {
            adaptiveDraftTokens = Swift.max(1, adaptiveDraftTokens - 1)
        }
    }

    private mutating func acceptMainOnly(lastBonusToken: MLXArray) {
        let output = targetModel.speculativeTargetOutput(
            .init(tokens: lastBonusToken.reshaped([1])),
            cache: targetCache.isEmpty ? nil : targetCache,
            state: targetState
        )
        targetState = output.state
        let token = sampleNextToken(logits: output.logits, processor: &processor)
        eval(token)
        processor?.didSample(token: token)
        pendingTokens.append(token.item(Int.self))
        quantizeKVCache(&targetCache)
        kvOffset += 1
        sharedKVStates = output.sharedKVStates
        draftHiddenStates = lastDraftHidden(from: output.hiddenStates)
        self.lastBonusToken = token
    }

    private mutating func nextPendingToken() -> Int? {
        guard pendingIndex < pendingTokens.count else {
            return nil
        }

        let token = pendingTokens[pendingIndex]
        pendingIndex += 1
        tokenCount += 1
        return token
    }

    private func sampleNextToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        let logits = logits[0..., -1, 0...]
        return sampleToken(logits: logits, processor: &processor)
    }

    private func sampleToken(logits: MLXArray, processor: inout LogitProcessor?) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        return sampler.sample(logits: logits)
    }

    private func lastDraftHidden(from hiddenStates: MLXArray) -> MLXArray {
        let lastIndex = Swift.max(hiddenStates.dim(1) - 1, 0)
        let hidden = hiddenStates[0..., lastIndex ..< lastIndex + 1, 0...]
        return targetModel.speculativeDraftHidden(hidden)
    }

    private func cacheOffset(_ cache: [KVCache]) -> Int {
        cache.map(\.offset).max() ?? 0
    }

    private func trimmedSharedKVStates(
        _ states: [String: SharedKVState],
        rejectedCount: Int
    ) -> [String: SharedKVState] {
        guard rejectedCount > 0 else {
            return states
        }
        return states.mapValues { state in
            let sequenceAxis = state.keys.shape.count - 2
            let validLength = Swift.max(1, state.keys.shape[sequenceAxis] - rejectedCount)
            return SharedKVState(
                keys: state.keys[.ellipsis, ..<validLength, 0...],
                values: state.values[.ellipsis, ..<validLength, 0...],
                offset: Swift.max(0, state.offset - rejectedCount)
            )
        }
    }
}

internal enum TokenGenerationIterator: Sequence, IteratorProtocol {
    case standard(TokenIterator)
    case speculative(SpeculativeTokenIterator)
    case nativeMTP(NativeMTPTokenIterator)
    case sharedKVMTP(SharedKVMTPTokenIterator)

    internal var cacheForPromptReuse: PromptCacheReusableState {
        switch self {
        case .standard(let iterator):
            PromptCacheReusableState(cache: iterator.cache)
        case .speculative(let iterator):
            iterator.cacheForPromptReuse
        case .nativeMTP(let iterator):
            iterator.cacheForPromptReuse
        case .sharedKVMTP(let iterator):
            iterator.cacheForPromptReuse
        }
    }

    internal mutating func next() -> Int? {
        switch self {
        case .standard(var iterator):
            let token = iterator.next()
            self = .standard(iterator)
            return token

        case .speculative(var iterator):
            let token = iterator.next()
            self = .speculative(iterator)
            return token

        case .nativeMTP(var iterator):
            let token = iterator.next()
            self = .nativeMTP(iterator)
            return token

        case .sharedKVMTP(var iterator):
            let token = iterator.next()
            self = .sharedKVMTP(iterator)
            return token
        }
    }
}

/// Result of a call to ``generate(input:parameters:context:didGenerate:)``.
///
/// This type is marked `@unchecked Sendable` because:
/// - It contains `LMInput.Text` which includes `MLXArray` (NOT inherently Sendable)
/// - However, the struct is immutable after creation (all properties are `let` constants)
/// - It represents a completed generation result that won't be modified
/// - The MLX arrays it contains are already evaluated and won't change
///
/// Safety guarantees:
/// - Immutable after creation: All properties are `let` constants
/// - No shared mutable state: Contains only read-only data
/// - Safe to pass across threads: The result is a snapshot of completed work
/// - Arrays are evaluated: MLX arrays are fully computed before being stored
internal struct GenerateResult: @unchecked Sendable {

    /// Initializes a new `GenerateResult` instance.
    ///
    /// - Parameters:
    ///   - inputText: The input text used for generation.
    ///   - tokens: The array of tokens generated.
    ///   - output: The generated output string.
    ///   - promptTime: The time taken to prompt the input.
    ///   - generateTime: The time taken to generate the output.
    internal init(
        inputText: LMInput.Text, tokens: [Int], output: String, promptTime: TimeInterval,
        generateTime: TimeInterval
    ) {
        self.inputText = inputText
        self.tokens = tokens
        self.output = output
        self.promptTime = promptTime
        self.generateTime = generateTime
    }

    /// input (prompt, images, etc.)
    internal let inputText: LMInput.Text

    @available(*, deprecated, message: "use inputText")
    internal var promptTokens: [Int] {
        inputText.tokens.asArray(Int.self)
    }

    /// output tokens
    internal let tokens: [Int]

    /// output text
    internal let output: String

    /// The number of tokens included in the input prompt.
    internal var promptTokenCount: Int { inputText.tokens.size }

    /// The number of tokens generated by the language model.
    internal var generationTokenCount: Int { tokens.count }

    /// time to process the prompt / generate the first token
    internal let promptTime: TimeInterval

    /// time to generate the remaining tokens
    internal let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    internal var promptTokensPerSecond: Double {
        Double(inputText.tokens.size) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    internal var tokensPerSecond: Double {
        Double(tokens.count) / generateTime
    }

    internal func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Action from token visitor callback in ``generate(input:parameters:context:didGenerate:)``.
internal enum GenerateDisposition: Sendable {
    /// keep producing tokens until an EOS token is produced
    case more

    /// stop producing tokens, e.g. a token limit has been hit
    case stop
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// For example:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let prompt: String
/// let context: ModelContext
///
/// let lmInput = try context.tokenize(prompt: prompt)
/// let result = generate(input: lmInput,
///     parameters: generateParameters,
///     context: context) { tokens in
///     .more
/// }
/// ```
///
/// Internally this constructs a ``TokenIterator`` and calls
/// ``generate(input:context:iterator:didGenerate:)``
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
internal func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: ([Int]) -> GenerateDisposition
) throws -> GenerateResult {
    let iterator = try TokenIterator(
        input: input, model: context.model, parameters: parameters)
    return generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

/// Low level token generation using a ``TokenIterator``.
///
/// ``generate(input:parameters:context:didGenerate:)`` is the preferred call.
///
/// - Parameters:
///   - input: prepared language model input
///   - context: model context (model and tokenizer)
///   - iterator: token iterator
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: the generated output
internal func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: ([Int]) -> GenerateDisposition
) -> GenerateResult {
    TextGenerator.generate(input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

extension TextGenerator {
    internal static func generate(
        input: LMInput, context: ModelContext,
        iterator: TokenIterator,
        didGenerate: ([Int]) -> GenerateDisposition
    ) -> GenerateResult {
    var start = Date.timeIntervalSinceReferenceDate
    var promptTime: TimeInterval = 0

    var additionalEOSTokenIds = context.configuration.eosTokenIds
    additionalEOSTokenIds.formUnion(
        context.configuration.extraEOSTokens
            .compactMap {
                context.tokenizer.convertTokenToId($0)
            })

    var tokens = [Int]()

    for token in iterator {
        // compute the timing for the prompt
        if tokens.isEmpty {
            let now = Date.timeIntervalSinceReferenceDate
            promptTime = now - start
            start = now
        }

        if token == context.tokenizer.unknownTokenId || token == context.tokenizer.eosTokenId
            || additionalEOSTokenIds.contains(token)
        {
            break
        }
        tokens.append(token)

        if didGenerate(tokens) == .stop {
            break
        }
    }

    let now = Date.timeIntervalSinceReferenceDate
    let generateTime = now - start

    // TokenIterator uses `asyncEval()` to keep the pipeline full. If the caller
    // exits the program right away, those tasks will still be executing and will
    // hit assertions as the mlx scheduler is torn down. Synchronize with the stream
    // to make sure it is complete.
    Stream().synchronize()

    return GenerateResult(
        inputText: input.text, tokens: tokens,
        output: context.tokenizer.decode(tokens: tokens),
        promptTime: promptTime, generateTime: generateTime)
    }
}

/// Generate tokens from an ``LMInput`` and a ``ModelContext``.
///
/// For example:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let prompt: String
/// let context: ModelContext
///
/// let lmInput = try context.tokenize(prompt: prompt)
/// let result = generate(input: lmInput,
///     parameters: generateParameters,
///     context: context) { token in
///     .more
/// }
/// ```
///
/// Internally this constructs a ``TokenIterator`` and calls
/// ``generate(input:context:iterator:didGenerate:)``
///
/// - Parameters:
///   - input: prepared language model input
///   - parameters: parameters controlling the token generation
///   - context: model context (model and tokenizer)
///   - didGenerate: token visitor that can output tokens as they are generated and indicate early stop
/// - Returns: Information about the generation
internal func generate(
    input: LMInput, parameters: GenerateParameters, context: ModelContext,
    didGenerate: (Int) -> GenerateDisposition
) throws -> GenerateCompletionInfo {
    try TextGenerator.generate(
        input: input, parameters: parameters, context: context, didGenerate: didGenerate)
}

internal func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator,
    didGenerate: (Int) -> GenerateDisposition
) -> GenerateCompletionInfo {
    TextGenerator.generate(
        input: input, context: context, iterator: iterator, didGenerate: didGenerate)
}

extension TextGenerator {
    internal static func generate(
        input: LMInput, parameters: GenerateParameters, context: ModelContext,
        didGenerate: (Int) -> GenerateDisposition
    ) throws -> GenerateCompletionInfo {
        let iterator = try TokenIterator(
            input: input, model: context.model, parameters: parameters)
        return generate(
            input: input, context: context, iterator: iterator, didGenerate: didGenerate)
    }

    internal static func generate(
        input: LMInput, context: ModelContext,
        iterator: TokenIterator,
        didGenerate: (Int) -> GenerateDisposition
    ) -> GenerateCompletionInfo {
        var start = Date.timeIntervalSinceReferenceDate
        var promptTime: TimeInterval = 0

        var additionalEOSTokenIds = context.configuration.eosTokenIds
        additionalEOSTokenIds.formUnion(
            context.configuration.extraEOSTokens
                .compactMap {
                    context.tokenizer.convertTokenToId($0)
                })

        var tokenCount = 0

        for token in iterator {
            // Compute the timing for the prompt
            if promptTime == 0 {
                let now = Date.timeIntervalSinceReferenceDate
                promptTime = now - start
                start = now
            }

            // Check for end-of-sequence tokens
            if token == context.tokenizer.unknownTokenId || token == context.tokenizer.eosTokenId
                || additionalEOSTokenIds.contains(token)
            {
                break
            }

            tokenCount += 1

            // Invoke the callback with the current token
            if didGenerate(token) == .stop {
                break
            }
        }

        let now = Date.timeIntervalSinceReferenceDate
        let generateTime = now - start

        // Synchronize with the stream to ensure tasks are completed
        Stream().synchronize()

        return GenerateCompletionInfo(
            promptTokenCount: input.text.tokens.size,
            generationTokenCount: tokenCount,
            promptTime: promptTime,
            generationTime: generateTime
        )
    }
}

/// Generates tokens asynchronously using the provided language model input, parameters, and context.
///
/// This function initializes a `TokenIterator` with the given input, model, and generation parameters,
/// and then streams the token generation process via an `AsyncStream`. The resulting stream yields
/// instances of the `Generation` enum, which can represent either individual tokens or summary
/// completion information.
///
/// - Parameters:
///   - input: The input for the language model.
///   - cache: optional ``KVCache``
///   - parameters: The configuration options for token generation.
///   - context: The model context, including the model itself and associated tokenizer.
/// - Returns: An `AsyncStream` that emits `Generation` values, including generated tokens (`.token`)
///   and completion information (`.info`).
/// - Throws: An error if the `TokenIterator` initialization fails due to invalid input or model configuration.
///
/// ### Example Usage:
/// ```swift
/// // Define the input, parameters, and context for token generation.
/// let generateParameters: GenerateParameters
/// let prompt: String
/// let context: ModelContext
///
/// let lmInput = try context.tokenize(prompt: prompt)
///
/// // Call the generate function to get an AsyncStream.
/// let stream = try generate(input: lmInput, parameters: parameters, context: context)
///
/// // Process the stream asynchronously to handle generated tokens and completion info.
/// for await generation in stream {
///     switch generation {
///     case .token(let token):
///         print("Generated token: \(context.tokenizer.decode(tokens: [token])")
///     case .info(let info):
///         print("Finished: \(info.tokensPerSecond) tokens/s.")
///     }
/// }
/// ```
internal func generate(
    input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext
) throws -> AsyncStream<Generation> {
    try TextGenerator.generate(
        input: input, cache: cache, parameters: parameters, context: context)
}

internal func generate(
    input: LMInput, context: ModelContext,
    iterator: TokenIterator
) -> AsyncStream<Generation> {
    TextGenerator.generate(
        input: input, context: context, iterator: iterator)
}

extension TextGenerator {
    internal static func generate(
        input: LMInput, cache: [KVCache]? = nil, parameters: GenerateParameters, context: ModelContext
    ) throws -> AsyncStream<Generation> {
        let iterator = try TokenIterator(
            input: input, model: context.model, cache: cache, parameters: parameters)
        return generate(
            input: input, context: context, iterator: iterator)
    }

    internal static func generate(
        input: LMInput, context: ModelContext,
        iterator: TokenIterator
    ) -> AsyncStream<Generation> {

        AsyncStream { continuation in

            // Launch a Task to perform iteration asynchronously.
            let task = Task {
                var start = Date.timeIntervalSinceReferenceDate
                var promptTime: TimeInterval = 0

                var additionalEOSTokenIds = context.configuration.eosTokenIds
                additionalEOSTokenIds.formUnion(
                    context.configuration.extraEOSTokens
                        .compactMap {
                            context.tokenizer.convertTokenToId($0)
                        })

                var tokenCount = 0
                var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)

                for token in iterator {

                    // Check for cancellation on every loop iteration.
                    if Task.isCancelled { break }

                    if promptTime == 0 {
                        let now = Date.timeIntervalSinceReferenceDate
                        promptTime = now - start
                        start = now
                    }

                    if token == context.tokenizer.unknownTokenId
                        || token == context.tokenizer.eosTokenId
                        || additionalEOSTokenIds.contains(token)
                    {
                        break
                    }

                    detokenizer.append(token: token)
                    if let chunk = detokenizer.next() {
                        tokenCount += 1
                        continuation.yield(.chunk(chunk))
                    }
                }

                let now = Date.timeIntervalSinceReferenceDate
                let generateTime = now - start

                let info = GenerateCompletionInfo(
                    promptTokenCount: input.text.tokens.size,
                    generationTokenCount: tokenCount,
                    promptTime: promptTime,
                    generationTime: generateTime
                )
                continuation.yield(.info(info))

                // Synchronize with the stream to ensure tasks are completed
                Stream().synchronize()

                // Finalize the stream
                continuation.finish()
            }
            // When the consumer cancels (or ends) the stream,
            // cancel our underlying task.
            continuation.onTermination = { _ in
                task.cancel()
            }
        }
    }
}

/// Represents metadata and statistics related to token generation.
///
/// Provides information about the number of tokens processed during both the prompt and generation phases, as well as the time taken for each phase.
internal struct GenerateCompletionInfo: Sendable {
    /// The number of tokens included in the input prompt.
    internal let promptTokenCount: Int

    /// The number of tokens generated by the language model.
    internal let generationTokenCount: Int

    /// The time interval (in seconds) taken to process the input prompt.
    internal let promptTime: TimeInterval

    /// The time interval (in seconds) taken to generate the output tokens.
    internal let generateTime: TimeInterval

    /// The number of tokens processed per second during the prompt phase.
    internal var promptTokensPerSecond: Double {
        Double(promptTokenCount) / promptTime
    }

    /// The number of tokens generated per second during the generation phase.
    internal var tokensPerSecond: Double {
        Double(generationTokenCount) / generateTime
    }

    internal init(
        promptTokenCount: Int,
        generationTokenCount: Int,
        promptTime: TimeInterval,
        generationTime: TimeInterval
    ) {
        self.promptTokenCount = promptTokenCount
        self.generationTokenCount = generationTokenCount
        self.promptTime = promptTime
        self.generateTime = generationTime
    }

    internal func summary() -> String {
        """
        Prompt:     \(promptTokenCount) tokens, \(promptTokensPerSecond.formatted()) tokens/s
        Generation: \(generationTokenCount) tokens, \(tokensPerSecond.formatted()) tokens/s, \(generateTime.formatted())s
        """
    }
}

/// Represents the different stages or outputs of the token generation process.
///
/// This enum distinguishes between the following:
/// - `.chunk`: A decoded string from one or more tokens generated by the language model.
/// - `.info`: Metadata and performance statistics about the generation process.
internal enum Generation: Sendable {
    /// A generated token represented as a String
    case chunk(String)

    /// Completion information summarizing token counts and performance metrics.
    case info(GenerateCompletionInfo)

    /// Generated text or nil
    internal var chunk: String? {
        switch self {
        case .chunk(let string): string
        case .info: nil
        }
    }

    /// Completion info or nil
    internal var info: GenerateCompletionInfo? {
        switch self {
        case .chunk: nil
        case .info(let info): info
        }
    }

    /// Reducer that can be used with `throttle()` to gather elements into a batch
    @Sendable
    internal static func collect(_ batch: [Generation]?, _ element: Generation) -> [Generation] {
        (batch ?? []) + [element]
    }
}
