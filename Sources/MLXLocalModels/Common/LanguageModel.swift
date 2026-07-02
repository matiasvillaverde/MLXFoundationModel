import MLX
import MLXNN

/// Time, height, and width metadata for image or video frames.
internal struct THW: Equatable, Sendable {
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

    internal var product: Int {
        t * h * w
    }
}

/// Tokenized model input plus optional prepared media.
internal struct LMInput: Sendable {
    internal let text: Text
    internal let image: ProcessedImage?
    internal let video: ProcessedVideo?

    internal struct Text: @unchecked Sendable {
        internal let tokens: MLXArray
        internal let mask: MLXArray?

        internal init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        internal subscript(
            indices: MLXArrayIndex...,
            stream stream: StreamOrDevice = .default
        ) -> Text {
            slice(indices, includeMask: true, stream: stream)
        }

        internal subscript(
            text indices: MLXArrayIndex...,
            stream stream: StreamOrDevice = .default
        ) -> Text {
            slice(indices, includeMask: false, stream: stream)
        }

        private func slice(
            _ indices: [MLXArrayIndex],
            includeMask: Bool,
            stream: StreamOrDevice
        ) -> Text {
            Text(
                tokens: tokens[indices, stream: stream],
                mask: includeMask ? mask?[indices, stream: stream] : mask
            )
        }
    }

    internal struct ProcessedImage: @unchecked Sendable {
        internal let pixels: MLXArray
        internal let frames: [THW]?

        internal init(pixels: MLXArray, frames: [THW]? = nil) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    internal struct ProcessedVideo: @unchecked Sendable {
        internal let pixels: MLXArray
        internal let frames: [THW]?

        internal init(pixels: MLXArray, frames: [THW]? = nil) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    internal init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    internal init(
        text: Text,
        image: ProcessedImage? = nil,
        video: ProcessedVideo? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
    }
}

internal struct LMOutput {
    internal let logits: MLXArray
    internal let state: State?

    internal struct State {
        internal let crossAttentionStates: MLXArray?

        internal init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    internal init(logits: MLXArray, state: State? = nil) {
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

internal enum PrepareResult {
    case tokens(LMInput.Text)
    case logits(LMOutput)
}

internal protocol LanguageModel: Module {
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    func newCache(parameters: GenerateParameters?) -> [KVCache]

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

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

@inline(__always)
internal func lastTokenHiddenState(_ hiddenStates: MLXArray) -> MLXArray {
    hiddenStates[0..., -1, 0...]
}

@inline(__always)
internal func greedyTokenOutput(logits: MLXArray, state: LMOutput.State?) -> GreedyTokenOutput {
    GreedyTokenOutput(token: argMax(logits, axis: -1), state: state)
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
    internal func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        LMOutput(logits: callAsFunction(input.tokens, cache: cache))
    }

    internal func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) is not implemented for \(Self.self)")
    }
}

extension LanguageModel {
    internal func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    internal func sanitize(weights: [String: MLXArray], metadata: [String: String])
        -> [String: MLXArray]
    {
        sanitize(weights: weights)
    }
}

internal protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    internal func newCache(parameters: GenerateParameters?) -> [KVCache] {
        LanguageModelCacheFactory.make(
            layerCount: kvHeads.count,
            parameters: parameters
        )
    }
}

internal enum LanguageModelCacheFactory {
    static func make(layerCount: Int, parameters: GenerateParameters?) -> [KVCache] {
        guard layerCount > 0 else {
            return []
        }

        return (0 ..< layerCount).map { _ in
            attentionCache(parameters: parameters)
        }
    }

    static func attentionCache(
        parameters: GenerateParameters?,
        defaultMaxSize: Int? = nil,
        keep: Int = GenerationConstants.rotatingCacheKeepTokens,
        step: Int? = nil
    ) -> KVCache {
        guard let maxSize = effectiveMaxSize(
            requestedMaxSize: parameters?.maxKVSize,
            defaultMaxSize: defaultMaxSize
        ) else {
            let cache = KVCacheSimple()
            if let step {
                cache.step = step
            }
            return cache
        }

        return RotatingKVCache(
            maxSize: maxSize,
            keep: keep,
            step: step ?? 256
        )
    }

    private static func effectiveMaxSize(
        requestedMaxSize: Int?,
        defaultMaxSize: Int?
    ) -> Int? {
        switch (requestedMaxSize, defaultMaxSize) {
        case let (requested?, fallback?):
            min(requested, fallback)
        case let (requested?, nil):
            requested
        case let (nil, fallback?):
            fallback
        case (nil, nil):
            nil
        }
    }
}
