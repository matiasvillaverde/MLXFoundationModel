import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("LLM model prefill")
struct LLMModelPrefillTests {
    @Test("generation defaults keep expected public behavior")
    func generationDefaultsKeepExpectedBehavior() {
        #expect(GenerationConstants.defaultPrefillStepSize == 512)
        #expect(GenerationConstants.defaultRepetitionContextSize == 20)
        #expect(GenerationConstants.defaultRepetitionPenaltyRange == 64)
        #expect(GenerationConstants.rotatingCacheKeepTokens == 4)
        #expect(GenerationConstants.defaultKVCacheGroupSize == 64)
        #expect(GenerationConstants.stopSequenceCheckWindowSize == 10)
    }

    @Test("returns the full prompt when it fits in the prefill window")
    func returnsFullPromptWhenItFits() throws {
        let model = RecordingLLMModel()

        let result = try model.prepare(
            LMInput(tokens: MLXArray([1, 2, 3])),
            cache: [],
            windowSize: 4
        )

        #expect(tokens(from: result) == [1, 2, 3])
        #expect(model.prefilledChunks.isEmpty)
    }

    @Test("prefills chunks until the tail fits in the requested window")
    func prefillsChunksUntilTailFits() throws {
        let model = RecordingLLMModel()

        let result = try model.prepare(
            LMInput(tokens: MLXArray([1, 2, 3, 4, 5])),
            cache: [],
            windowSize: 2
        )

        #expect(tokens(from: result) == [5])
        #expect(model.prefilledChunks == [[1, 2], [3, 4]])
        #expect(model.sawNilCache == [true, true])
    }

    @Test("uses the default prefill size when no window is provided")
    func usesDefaultPrefillSize() throws {
        let model = RecordingLLMModel()
        let prompt = Array(0 ... GenerationConstants.defaultPrefillStepSize)

        let result = try model.prepare(
            LMInput(tokens: MLXArray(prompt)),
            cache: [],
            windowSize: nil
        )

        #expect(tokens(from: result) == [GenerationConstants.defaultPrefillStepSize])
        #expect(model.prefilledChunks.count == 1)
        #expect(model.prefilledChunks[0] == Array(0 ..< GenerationConstants.defaultPrefillStepSize))
    }

    private func tokens(from result: PrepareResult) -> [Int] {
        switch result {
        case .tokens(let text):
            text.tokens.asArray(Int.self)

        case .logits:
            []
        }
    }

    private final class RecordingLLMModel: Module, LLMModel {
        private(set) var prefilledChunks: [[Int]] = []
        private(set) var sawNilCache: [Bool] = []

        deinit {
            // Required by the strict test lint profile.
        }

        // swiftlint:disable discouraged_optional_collection
        func callAsFunction(
            _ input: LMInput.Text,
            cache: [KVCache]?,
            state: LMOutput.State?
        ) -> LMOutput {
            prefilledChunks.append(input.tokens.reshaped(-1).asArray(Int.self))
            sawNilCache.append(cache == nil)
            return LMOutput(logits: MLXArray([Float(0)]).reshaped([1, 1, 1]), state: state)
        }
        // swiftlint:enable discouraged_optional_collection

        func newCache(parameters: GenerateParameters?) -> [KVCache] {
            []
        }
    }
}
