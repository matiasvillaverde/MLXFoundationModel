// swiftlint:disable discouraged_optional_collection
import MLX
@testable import MLXLocalModels
import MLXNN

final class ScriptedGreedyTokenModel: Module, GreedyTokenModel {
    private(set) var greedyStepCount = 0
    private(set) var fullLogitsStepCount = 0

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
        fullLogitsStepCount += 1
        return Self.logits(favoring: 3)
    }

    func greedyToken(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> GreedyTokenOutput {
        greedyStepCount += 1
        return GreedyTokenOutput(token: MLXArray([7]), state: state)
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private static func logits(favoring tokenID: Int) -> MLXArray {
        var row = Array(repeating: Float(-100), count: 8)
        row[tokenID] = 100
        return MLXArray(row).reshaped([1, 1, 8])
    }
}
// swiftlint:enable discouraged_optional_collection
