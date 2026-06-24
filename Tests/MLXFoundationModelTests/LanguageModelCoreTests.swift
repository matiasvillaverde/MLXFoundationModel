// swiftlint:disable discouraged_optional_collection
import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("Language model core")
struct LanguageModelCoreTests {
    @Test("THW exposes values and product")
    func thwExposesValuesAndProduct() {
        let dimensions = THW(2, 3, 5)

        #expect(dimensions.values == (2, 3, 5))
        #expect(dimensions.product == 30)
    }

    @Test("text slices tokens and masks together")
    func textSlicesTokensAndMasksTogether() {
        let text = LMInput.Text(
            tokens: MLXArray([1, 2, 3]),
            mask: MLXArray([10, 11, 12])
        )

        let sliced = text[0 ... 1]

        #expect(sliced.tokens.asArray(Int.self) == [1, 2])
        #expect(sliced.mask?.asArray(Int.self) == [10, 11])
    }

    @Test("text-only slice preserves original mask")
    func textOnlySlicePreservesOriginalMask() {
        let text = LMInput.Text(
            tokens: MLXArray([1, 2, 3]),
            mask: MLXArray([10, 11, 12])
        )

        let expanded = text[text: .newAxis]

        #expect(expanded.tokens.shape == [1, 3])
        #expect(expanded.tokens.asArray(Int.self) == [1, 2, 3])
        #expect(expanded.mask?.shape == [3])
        #expect(expanded.mask?.asArray(Int.self) == [10, 11, 12])
    }

    @Test("input can carry prepared image and video frames")
    func inputCanCarryPreparedMedia() {
        let imageFrames = [THW(1, 2, 3)]
        let videoFrames = [THW(4, 5, 6)]
        let input = LMInput(
            text: .init(tokens: MLXArray([7])),
            image: .init(pixels: MLXArray([Float(1)]), frames: imageFrames),
            video: .init(pixels: MLXArray([Float(2)]), frames: videoFrames)
        )

        #expect(input.text.tokens.asArray(Int.self) == [7])
        #expect(input.image?.frames == imageFrames)
        #expect(input.video?.frames == videoFrames)
    }

    @Test("default language model output forwards through array call")
    func defaultLanguageModelOutputForwardsThroughArrayCall() {
        let model = ForwardingLanguageModel(kvHeads: [1])
        let output = model(
            LMInput.Text(tokens: MLXArray([4, 5])),
            cache: nil,
            state: LMOutput.State(crossAttentionStates: MLXArray([Float(9)]))
        )

        #expect(model.forwardedTokens == [4, 5])
        #expect(output.logits.shape == [1, 1, 3])
        #expect(output.logits.asArray(Float.self) == [0, 3, 1])
        #expect(output.state == nil)
    }

    @Test("last-token helper returns final sequence row")
    func lastTokenHelperReturnsFinalSequenceRow() {
        let hiddenStates = MLXArray((1 ... 6).map(Float.init)).reshaped([1, 3, 2])

        let last = lastTokenHiddenState(hiddenStates)

        #expect(last.shape == [1, 2])
        #expect(last.asArray(Float.self) == [5, 6])
    }

    @Test("greedy token helper preserves state")
    func greedyTokenHelperPreservesState() throws {
        let state = LMOutput.State(crossAttentionStates: MLXArray([Float(3)]))
        let logits = MLXArray([Float(-1), Float(4), Float(2)]).reshaped([1, 1, 3])

        let output = greedyTokenOutput(logits: logits, state: state)

        #expect(output.token.item(Int.self) == 1)
        #expect(try #require(output.state?.crossAttentionStates).item(Float.self) == 3)
    }

    @Test("default sanitize keeps weights unchanged")
    func defaultSanitizeKeepsWeightsUnchanged() throws {
        let model = ForwardingLanguageModel(kvHeads: [])
        let weights = ["weight": MLXArray([Float(1), Float(2)])]

        let sanitized = model.sanitize(weights: weights, metadata: ["weight": "float32"])

        #expect(try #require(sanitized["weight"]).asArray(Float.self) == [1, 2])
    }

    @Test("dimension provider creates one cache per layer")
    func dimensionProviderCreatesOneCachePerLayer() throws {
        let model = ForwardingLanguageModel(kvHeads: [2, 4, 8])

        let simpleCaches = model.newCache(parameters: nil)
        let rotatingCaches = model.newCache(parameters: GenerateParameters(maxKVSize: 16))

        #expect(simpleCaches.count == 3)
        #expect(simpleCaches.allSatisfy { $0 is KVCacheSimple })
        #expect(rotatingCaches.count == 3)
        #expect(rotatingCaches.map(\.maxSize) == [16, 16, 16])
        #expect(try #require(rotatingCaches[0] as? RotatingKVCache) !==
            #require(rotatingCaches[1] as? RotatingKVCache))
    }

    private final class ForwardingLanguageModel: Module, LanguageModel, KVCacheDimensionProvider {
        let kvHeads: [Int]
        private(set) var forwardedTokens: [Int] = []

        init(kvHeads: [Int]) {
            self.kvHeads = kvHeads
            super.init()
        }

        deinit {
            // Required by the strict test lint profile.
        }

        func prepare(
            _ input: LMInput,
            cache: [KVCache],
            windowSize: Int?
        ) throws -> PrepareResult {
            .tokens(input.text)
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
            forwardedTokens = inputs.asArray(Int.self)
            return MLXArray([Float(0), Float(3), Float(1)]).reshaped([1, 1, 3])
        }
    }
}
// swiftlint:enable discouraged_optional_collection
