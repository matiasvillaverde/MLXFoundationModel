import MLX
@testable import MLXLocalModels
import MLXNN

final class MLXEchoBatchLanguageModel: Module, LanguageModel {
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

    // swiftlint:disable discouraged_optional_collection
    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray {
        let rowCount = inputs.dim(0)
        let vocabularySize = 3
        let values = (0 ..< rowCount).flatMap { row in
            Self.logits(row: row)
        }
        return MLXArray(values).reshaped([rowCount, 1, vocabularySize])
    }
    // swiftlint:enable discouraged_optional_collection

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
    }

    private static func logits(row: Int) -> [Float] {
        if row == 0 {
            return [0, 10, 9]
        }
        return [9, 8, 1]
    }
}
