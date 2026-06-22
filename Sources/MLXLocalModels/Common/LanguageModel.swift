// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import Tokenizers

/// Time/Height/Width struct to represent information about input images.
internal struct THW: Sendable {

    internal let t: Int
    internal let h: Int
    internal let w: Int

    internal init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    internal var values: (Int, Int, Int) {
        (t, h, w)
    }

    internal var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed by the model.
internal struct LMInput: Sendable {
    internal let text: Text
    internal let image: ProcessedImage?
    internal let video: ProcessedVideo?

    /// Representation of tokenized input text.
    ///
    /// This type is marked `@unchecked Sendable` because:
    /// - It contains `MLXArray` instances which are NOT inherently Sendable
    /// - However, the struct is immutable after creation (all properties are `let` constants)
    /// - It represents input that won't be modified during model execution
    /// - The MLX arrays are used in a controlled, single-threaded context (ModelContainer)
    ///
    /// Safety guarantees:
    /// - Immutable after creation: Both `tokens` and `mask` are `let` constants
    /// - Single-threaded usage: Used within ModelContainer which ensures serial GPU access
    /// - No shared mutable state: The arrays are not modified after creation
    /// - Controlled context: Only passed to model execution functions, not shared across threads
    internal struct Text: @unchecked Sendable {

        /// input token array
        internal let tokens: MLXArray

        /// optional mask array
        internal let mask: MLXArray?

        internal init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        internal subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        internal subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    /// Representation of prepared input image(s).
    ///
    /// This type is marked `@unchecked Sendable` because:
    /// - It contains `MLXArray` (pixels) which is NOT inherently Sendable
    /// - However, the struct is immutable after creation (all properties are `let` constants)
    /// - It represents preprocessed image data that won't be modified
    /// - The MLX array is used in a controlled, single-threaded context (ModelContainer)
    ///
    /// Safety guarantees:
    /// - Immutable after creation: Both `pixels` and `frames` are `let` constants
    /// - Single-threaded usage: Used within ModelContainer which ensures serial GPU access
    /// - No shared mutable state: The pixel array is not modified after preprocessing
    /// - Controlled context: Only passed to vision-language model execution
    internal struct ProcessedImage: @unchecked Sendable {

        /// Concatenated pixels from one or more images
        internal let pixels: MLXArray
        /// Time, height, and width of the images
        internal let frames: [THW]?

        internal init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    ///
    /// This type is marked `@unchecked Sendable` because:
    /// - It contains `MLXArray` (pixels) which is NOT inherently Sendable
    /// - However, the struct is immutable after creation (all properties are `let` constants)
    /// - It represents preprocessed video frames that won't be modified
    /// - The MLX array is used in a controlled, single-threaded context (ModelContainer)
    ///
    /// Safety guarantees:
    /// - Immutable after creation: Both `pixels` and `frames` are `let` constants
    /// - Single-threaded usage: Used within ModelContainer which ensures serial GPU access
    /// - No shared mutable state: The pixel array is not modified after preprocessing
    /// - Controlled context: Only passed to vision-language model execution
    internal struct ProcessedVideo: @unchecked Sendable {

        internal let pixels: MLXArray
        internal let frames: [THW]?

        internal init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    internal init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    internal init(
        text: LMInput.Text, image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
internal struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    internal let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    internal let state: State?

    internal struct State {
        internal let crossAttentionStates: MLXArray?

        internal init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    internal init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

internal struct NativeMTPMainOutput {
    internal let logits: MLXArray
    internal let hiddenStates: MLXArray
    internal let state: LMOutput.State?

    internal init(
        logits: MLXArray,
        hiddenStates: MLXArray,
        state: LMOutput.State? = nil
    ) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.state = state
    }
}

internal struct NativeMTPDraftOutput {
    internal let logits: MLXArray
    internal let hiddenStates: MLXArray

    internal init(logits: MLXArray, hiddenStates: MLXArray) {
        self.logits = logits
        self.hiddenStates = hiddenStates
    }
}

internal struct GreedyTokenOutput {
    internal let token: MLXArray
    internal let state: LMOutput.State?

    internal init(token: MLXArray, state: LMOutput.State? = nil) {
        self.token = token
        self.state = state
    }
}

internal struct SharedKVState {
    internal let keys: MLXArray
    internal let values: MLXArray
    internal let offset: Int

    internal init(keys: MLXArray, values: MLXArray, offset: Int) {
        self.keys = keys
        self.values = values
        self.offset = offset
    }
}

internal struct SharedKVTargetOutput {
    internal let logits: MLXArray
    internal let hiddenStates: MLXArray
    internal let sharedKVStates: [String: SharedKVState]
    internal let state: LMOutput.State?

    internal init(
        logits: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: SharedKVState],
        state: LMOutput.State? = nil
    ) {
        self.logits = logits
        self.hiddenStates = hiddenStates
        self.sharedKVStates = sharedKVStates
        self.state = state
    }
}

internal struct SharedKVDraftOutput {
    internal let logits: MLXArray
    internal let hiddenStates: MLXArray

    internal init(logits: MLXArray, hiddenStates: MLXArray) {
        self.logits = logits
        self.hiddenStates = hiddenStates
    }
}

internal struct SharedKVGreedyDraftOutput {
    internal let token: MLXArray
    internal let hiddenStates: MLXArray

    internal init(token: MLXArray, hiddenStates: MLXArray) {
        self.token = token
        self.hiddenStates = hiddenStates
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:windowSize:)``
internal enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
internal protocol LanguageModel: Module {

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// create a new array of ``KVCache`` -- automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]

    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with safetensors metadata.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

internal protocol NativeMTPModel: LanguageModel {
    var supportsNativeMTP: Bool { get }

    func makeMTPCache(parameters: GenerateParameters?) -> [KVCache]

    func nativeMTPMainOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> NativeMTPMainOutput

    func nativeMTPDraftOutput(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPDraftOutput?
}

internal protocol GreedyTokenModel: LanguageModel {
    func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput
}

internal protocol SharedKVSpeculativeTargetModel: LanguageModel {
    func speculativePrepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> SharedKVTargetOutput

    func speculativeTargetOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> SharedKVTargetOutput

    func speculativeDraftHidden(_ hiddenStates: MLXArray) -> MLXArray

    func speculativeTokenEmbeddings(_ tokenIDs: MLXArray) -> MLXArray
}

internal protocol SharedKVSpeculativeDraftModel: LanguageModel {
    var speculativeDraftBlockSize: Int { get }

    func sharedKVDraftOutput(
        tokenEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: SharedKVState],
        position: Int
    ) -> SharedKVDraftOutput

    func sharedKVGreedyDraftOutput(
        tokenEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: SharedKVState],
        position: Int
    ) -> SharedKVGreedyDraftOutput?
}

extension LanguageModel {
    internal func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

extension LanguageModel {
    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    internal func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        sanitize(weights: weights)
    }
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
internal protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if maxKVSize is provided
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(
                    maxSize: maxKVSize,
                    keep: GenerationConstants.rotatingCacheKeepTokens
                )
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
