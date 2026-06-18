// Copyright © 2024 Apple Inc.

import Foundation
@preconcurrency import MLX
import MLXNN
import Tokenizers

/// Namespace for text generation functions to avoid naming conflicts
internal enum TextGenerator {}

/// A `LogitSampler` is responsible for sampling `logits` produced by
/// a ``LanguageModel`` to produce a token.
///
/// See also: ``LogitProcessor``
internal protocol LogitSampler: Sendable {

    /// Given `logits` produce a new `MLXArray` with the token.
    func sample(logits: MLXArray) -> MLXArray
}

/// A `LogitProcessor` is an optional visitor of `logits`.
///
/// The ``LogitProcessor`` is called with the input (prompt) before generating tokens:
///
/// ```swift
/// processor?.prompt(input.text.tokens)
/// ```
///
/// Then for each token generated it has a chance to adjust the logits:
///
/// ```swift
/// logits = processor?.process(logits: logits) ?? logits
/// let y = sampler.sample(logits: logits)
/// processor?.didSample(token: y)
/// ```
///
/// See also: ``LogitSampler``
internal protocol LogitProcessor: Sendable {

    /// called before token generation starts with the text tokens of the prompt
    mutating func prompt(_ prompt: MLXArray)

    /// called to visit and possibly modify the logits
    mutating func process(logits: MLXArray) -> MLXArray

    /// called to provide the sampled token
    mutating func didSample(token: MLXArray)
}

/// Parameters for text generation, see ``TokenIterator``.
///
/// This produces:
///
/// - ``LogitSampler``
/// - ``LogitProcessor``
///
/// for the `TokenIterator`.
internal struct GenerateParameters: Sendable {

    /// Step size for processing the prompt
    internal var prefillStepSize: Int

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

    /// sampling temperature
    internal var temperature: Float = 0.6

    /// top p sampling
    internal var topP: Float = 1.0

    /// Top-k sampling. A value of 0 disables this filter.
    internal var topK: Int = 0

    /// Min-p sampling threshold relative to the highest-probability token.
    internal var minP: Float = 0.0

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

    /// Token-level grammar constraint.
    internal var grammar: GrammarSamplingConfiguration?

    internal init(
        maxTokens: Int? = nil,
        maxKVSize: Int? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = GenerationConstants.defaultKVCacheGroupSize,
        quantizedKVStart: Int = 0,
        temperature: Float = 0.6,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        repetitionPenalty: Float? = nil,
        repetitionContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        presencePenalty: Float? = nil,
        presenceContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        frequencyPenalty: Float? = nil,
        frequencyContextSize: Int = GenerationConstants.defaultRepetitionContextSize,
        seed: Int? = nil,
        grammar: GrammarSamplingConfiguration? = nil,
        logitBias: [Int: Float] = [:],
        prefillStepSize: Int = GenerationConstants.defaultPrefillStepSize
    ) {
        self.maxTokens = maxTokens
        self.maxKVSize = maxKVSize
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.repetitionContextSize = repetitionContextSize
        self.presencePenalty = presencePenalty
        self.presenceContextSize = presenceContextSize
        self.frequencyPenalty = frequencyPenalty
        self.frequencyContextSize = frequencyContextSize
        self.seed = seed
        self.grammar = grammar
        self.logitBias = logitBias
        self.prefillStepSize = prefillStepSize
    }

    internal func sampler() -> LogitSampler {
        let usesTopP = topP > 0 && topP < 1
        let usesTopK = topK > 0
        let usesMinP = minP > 0

        if temperature == 0 {
            return ArgMaxSampler()
        } else if usesTopP || usesTopK || usesMinP {
            return TopPSampler(
                temperature: temperature,
                topP: topP,
                topK: topK,
                minP: minP,
                seed: seed
            )
        } else {
            return CategoricalSampler(temperature: temperature, seed: seed)
        }
    }

    internal func processor(
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws -> LogitProcessor? {
        let repetitionContext: RepetitionContext?
        if let repetitionPenalty, repetitionPenalty != 0, repetitionPenalty != 1,
           repetitionContextSize > 0
        {
            repetitionContext = RepetitionContext(
                repetitionPenalty: repetitionPenalty,
                repetitionContextSize: repetitionContextSize
            )
        } else {
            repetitionContext = nil
        }

        let presenceContext: PresencePenaltyContext?
        if let presencePenalty, presencePenalty != 0, presenceContextSize > 0 {
            presenceContext = PresencePenaltyContext(
                presencePenalty: presencePenalty,
                presenceContextSize: presenceContextSize
            )
        } else {
            presenceContext = nil
        }

        let frequencyContext: FrequencyPenaltyContext?
        if let frequencyPenalty, frequencyPenalty != 0, frequencyContextSize > 0 {
            frequencyContext = FrequencyPenaltyContext(
                frequencyPenalty: frequencyPenalty,
                frequencyContextSize: frequencyContextSize
            )
        } else {
            frequencyContext = nil
        }

        let logitBiasContext = logitBias.isEmpty ? nil : LogitBiasProcessor(logitBias: logitBias)
        let grammarContext: GrammarConstrainedLogitProcessor?
        if let grammar {
            guard let grammarCompiler else {
                throw GrammarConstraintError.missingGrammarCompiler
            }
            grammarContext = try GrammarConstrainedLogitProcessor(
                matcher: grammarCompiler.makeMatcher(for: grammar)
            )
        } else {
            grammarContext = nil
        }

        if repetitionContext == nil && presenceContext == nil && frequencyContext == nil &&
            logitBiasContext == nil && grammarContext == nil
        {
            return nil
        }

        return PenaltyProcessor(
            logitBiasContext: logitBiasContext,
            repetitionContext: repetitionContext,
            presenceContext: presenceContext,
            frequencyContext: frequencyContext,
            grammarContext: grammarContext
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
internal struct TopPSampler: LogitSampler, @unchecked Sendable {
    let temp: MLXArray
    let topP: MLXArray?
    let topK: Int?
    let minP: MLXArray?
    let negInf: MLXArray
    let randomState: MLXRandom.RandomState

    internal init(
        temperature: Float,
        topP: Float = 1.0,
        topK: Int = 0,
        minP: Float = 0.0,
        seed: Int? = nil
    ) {
        self.temp = MLXArray(temperature)
        self.topP = topP > 0 && topP < 1 ? MLXArray(topP) : nil
        self.topK = topK > 0 ? topK : nil
        self.minP = minP > 0 ? MLXArray(minP) : nil
        self.negInf = MLXArray(-Float.infinity)
        self.randomState = makeRandomState(seed: seed)
    }

    internal func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        if logits.dtype == .bfloat16 {
            logits = logits.asType(.float32)
        }

        return withRandomState(randomState) {
            var logprobs = logSoftmax(logits)

            // Match Python mlx-lm filter order: top-p, min-p, then top-k.
            if let topP {
                logprobs = applyTopP(logprobs, topP: topP)
            }
            if let minP {
                logprobs = applyMinP(logprobs, minP: minP)
            }
            if let topK {
                logprobs = applyTopK(logprobs, topK: topK)
            }

            return categorical(logprobs * (1 / temp))
        }
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

    private func applyTopK(_ logprobs: MLXArray, topK: Int) -> MLXArray {
        let vocabularySize = logprobs.dim(-1)
        guard topK < vocabularySize else { return logprobs }

        let maskIndices = argPartition(-logprobs, kth: topK - 1, axis: -1)[0..., topK...]
        return putAlong(logprobs, maskIndices, values: negInf, axis: -1)
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

/// Composes active penalty processors in the same order as Python mlx-lm.
internal struct PenaltyProcessor: LogitProcessor, @unchecked Sendable {
    var logitBiasContext: LogitBiasProcessor?
    var repetitionContext: RepetitionContext?
    var presenceContext: PresencePenaltyContext?
    var frequencyContext: FrequencyPenaltyContext?
    var grammarContext: GrammarConstrainedLogitProcessor?

    internal init(
        logitBiasContext: LogitBiasProcessor?,
        repetitionContext: RepetitionContext?,
        presenceContext: PresencePenaltyContext?,
        frequencyContext: FrequencyPenaltyContext?,
        grammarContext: GrammarConstrainedLogitProcessor?
    ) {
        self.logitBiasContext = logitBiasContext
        self.repetitionContext = repetitionContext
        self.presenceContext = presenceContext
        self.frequencyContext = frequencyContext
        self.grammarContext = grammarContext
    }

    mutating internal func prompt(_ prompt: MLXArray) {
        logitBiasContext?.prompt(prompt)
        repetitionContext?.prompt(prompt)
        presenceContext?.prompt(prompt)
        frequencyContext?.prompt(prompt)
        grammarContext?.prompt(prompt)
    }

    mutating internal func process(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = logitBiasContext?.process(logits: logits) ?? logits
        logits = repetitionContext?.process(logits: logits) ?? logits
        logits = presenceContext?.process(logits: logits) ?? logits
        logits = frequencyContext?.process(logits: logits) ?? logits
        logits = grammarContext?.process(logits: logits) ?? logits

        return logits
    }

    mutating internal func didSample(token: MLXArray) {
        logitBiasContext?.didSample(token: token)
        repetitionContext?.didSample(token: token)
        presenceContext?.didSample(token: token)
        frequencyContext?.didSample(token: token)
        grammarContext?.didSample(token: token)
    }
}

/// Generator of tokens.
///
/// This is typically used via a call to ``generate(input:parameters:context:didGenerate:)``.
///
/// To use it directly:
///
/// ```swift
/// let generateParameters: GenerateParameters
/// let input: LMInput
/// let model: LanguageModel
///
/// let iterator = try TokenIterator(input: input, model: model, parameters: parameters)
///
/// for token in iterator {
///     ...
/// }
/// ```
///
/// Tokens are integers that can be passed through a `Tokenizer` or ``StreamingDetokenizer`` to produce Strings.
///
/// Port of `generate_step()` from https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py
///
/// Note: this uses `asyncEval()` and there may be an async evaluation running after a call to `next()`.
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
        let result = model(
            previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        self.state = result.state

        // Apply dynamic cache quantization after each step
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart
        )
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "step", cache: cache)

        return convertToToken(logits: result.logits)
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
        self.quantizeKVCache = { cache in
            maybeQuantizeKVCache(
                cache: &cache,
                kvBits: parameters.kvBits,
                kvGroupSize: parameters.kvGroupSize,
                quantizedKVStart: parameters.quantizedKVStart
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
        let remainingTokens = maxTokens.map { $0 - tokenCount } ?? numDraftTokens
        let draftCount = Swift.min(remainingTokens, numDraftTokens)
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

        trimPromptCache(mainCache, numTokens: draftTokens.count - acceptedCount)
        trimPromptCache(draftCache, numTokens: Swift.max(draftTokens.count - acceptedCount - 1, 0))
        quantizeKVCache(&mainCache)
        quantizeKVCache(&draftCache)

        y = .init(tokens: finalToken)
        draftY = .init(tokens: finalToken)
        if acceptedCount == draftTokens.count, let lastDraftToken = draftTokens.last {
            draftY = .init(tokens: concatenated([lastDraftToken.reshaped([1]), finalToken]))
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

internal enum TokenGenerationIterator: Sequence, IteratorProtocol {
    case standard(TokenIterator)
    case speculative(SpeculativeTokenIterator)

    internal var cacheForPromptReuse: PromptCacheReusableState {
        switch self {
        case .standard(let iterator):
            PromptCacheReusableState(cache: iterator.cache)
        case .speculative(let iterator):
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
