// swiftlint:disable discouraged_optional_collection
import MLX
@testable import MLXLocalModels
import MLXNN

final class ScriptedNativeMTPModel: Module, NativeMTPModel {
    enum Verification {
        case accept
        case reject
    }

    let mainCache = ScriptedNativeMTPCache()
    let mtpCache = ScriptedNativeMTPCache()
    private let verification: Verification
    private(set) var mainStepCount = 0
    private(set) var verifyStepCount = 0
    private(set) var draftStepCount = 0

    init(verification: Verification) {
        self.verification = verification
        super.init()
    }

    deinit {
        // Required by the strict test lint profile.
    }

    var supportsNativeMTP: Bool {
        true
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        verifyStepCount += 1
        advance(cache, by: inputs.dim(1))

        switch verification {
        case .accept:
            return Self.logits(favoring: [2, 3])

        case .reject:
            return Self.logits(favoring: [4, 3])
        }
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [mainCache]
    }

    func makeMTPCache(parameters: GenerateParameters?) -> [KVCache] {
        [mtpCache]
    }

    func nativeMTPMainOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> NativeMTPMainOutput {
        mainStepCount += 1
        advance(cache, by: input.tokens.dim(1))
        return NativeMTPMainOutput(
            logits: Self.logits(favoring: [1]),
            hiddenStates: MLXArray.ones([1, input.tokens.dim(1), 2])
        )
    }

    func nativeMTPDraftOutput(
        hiddenStates: MLXArray,
        nextTokenIDs: MLXArray,
        cache: [KVCache]?
    ) -> NativeMTPDraftOutput? {
        draftStepCount += 1
        advance(cache, by: nextTokenIDs.dim(1))
        return NativeMTPDraftOutput(
            logits: Self.logits(favoring: [2]),
            hiddenStates: MLXArray.ones([1, 1, 2])
        )
    }

    private func advance(_ cache: [KVCache]?, by tokenCount: Int) {
        cache?.forEach { cache in
            (cache as? ScriptedNativeMTPCache)?.advance(by: tokenCount)
        }
    }

    private static func logits(favoring tokenIDs: [Int]) -> MLXArray {
        let values = tokenIDs.flatMap { tokenID in
            Self.row(favoring: tokenID)
        }
        return MLXArray(values).reshaped([1, tokenIDs.count, Self.vocabularySize])
    }

    private static func row(favoring tokenID: Int) -> [Float] {
        var row = Array(repeating: Float(-100), count: vocabularySize)
        row[tokenID] = 100
        return row
    }

    private static let vocabularySize = 8
}
// swiftlint:enable discouraged_optional_collection
