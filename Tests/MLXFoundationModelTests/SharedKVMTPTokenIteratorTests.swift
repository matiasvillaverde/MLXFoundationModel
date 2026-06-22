import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

// swiftlint:disable file_types_order
@Suite("Shared-KV MTP token iterator")
struct SharedKVMTPTokenIteratorTests {
    @Test("accepts matching shared-KV draft token")
    func acceptsMatchingSharedKVDraftToken() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ScriptedSharedKVTargetModel(verification: .accept)
            let draft = ScriptedSharedKVDraftModel()
            var iterator = try SharedKVMTPTokenIterator(
                input: LMInput(tokens: MLXArray([Int32(10)])),
                targetModel: target,
                draftModel: draft,
                parameters: Self.parameters,
                numDraftTokens: 1
            )

            #expect(iterator.next() == 1)
            #expect(iterator.next() == 2)
            #expect(iterator.next() == 3)
            #expect(target.mainCache.offset == 3)
            #expect(target.prepareStepCount == 1)
            #expect(target.verifyStepCount == 1)
            #expect(draft.draftStepCount == 1)
        }
    }

    @Test("trims rejected shared-KV draft token")
    func trimsRejectedSharedKVDraftToken() throws {
        try Device.withDefaultDevice(.cpu) {
            let target = ScriptedSharedKVTargetModel(verification: .reject)
            let draft = ScriptedSharedKVDraftModel()
            var iterator = try SharedKVMTPTokenIterator(
                input: LMInput(tokens: MLXArray([Int32(10)])),
                targetModel: target,
                draftModel: draft,
                parameters: Self.parameters,
                numDraftTokens: 1
            )

            #expect(iterator.next() == 1)
            #expect(iterator.next() == 4)
            #expect(target.mainCache.offset == 2)
            #expect(target.mainCache.trimmedTokenCount == 1)
            #expect(draft.draftStepCount == 1)
        }
    }

    private static var parameters: GenerateParameters {
        GenerateParameters(maxTokens: 3, temperature: 0)
    }
}
// swiftlint:enable file_types_order

// swiftlint:disable one_declaration_per_file
final class ScriptedSharedKVTargetModel: Module, SharedKVSpeculativeTargetModel {
    enum Verification {
        case accept
        case reject
    }

    let mainCache = ScriptedNativeMTPCache()
    private let verification: Verification
    private(set) var prepareStepCount = 0
    private(set) var verifyStepCount = 0

    init(verification: Verification) {
        self.verification = verification
        super.init()
    }

    deinit {
        // Required by the strict test lint profile.
    }

    func speculativePrepare(
        _ input: LMInput,
        cache: [KVCache],
        windowSize: Int?
    ) throws -> SharedKVTargetOutput {
        prepareStepCount += 1
        advance(cache, by: input.text.tokens.dim(0))
        return SharedKVTargetOutput(
            logits: Self.logits(favoring: [1]),
            hiddenStates: MLXArray.ones([1, input.text.tokens.dim(0), 2]),
            sharedKVStates: sharedKVStates()
        )
    }

    func speculativeTargetOutput(
        _ input: LMInput.Text,
        cache: [KVCache]?, // swiftlint:disable:this discouraged_optional_collection
        state: LMOutput.State?
    ) -> SharedKVTargetOutput {
        verifyStepCount += 1
        advance(cache, by: input.tokens.dim(0))

        let favoredTokens: [Int]
        switch verification {
        case .accept:
            favoredTokens = [2, 3]

        case .reject:
            favoredTokens = [4, 3]
        }
        return SharedKVTargetOutput(
            logits: Self.logits(favoring: favoredTokens),
            hiddenStates: MLXArray.ones([1, input.tokens.dim(0), 2]),
            sharedKVStates: sharedKVStates()
        )
    }

    func speculativeDraftHidden(_ hiddenStates: MLXArray) -> MLXArray {
        hiddenStates
    }

    func speculativeTokenEmbeddings(_ tokenIDs: MLXArray) -> MLXArray {
        MLXArray.ones(Array(tokenIDs.shape) + [2])
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? // swiftlint:disable:this discouraged_optional_collection
    ) -> MLXArray {
        Self.logits(favoring: [1])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [mainCache]
    }

    private func advance(
        _ cache: [KVCache]?, // swiftlint:disable:this discouraged_optional_collection
        by tokenCount: Int
    ) {
        cache?.forEach { cache in
            (cache as? ScriptedNativeMTPCache)?.advance(by: tokenCount)
        }
    }

    private func sharedKVStates() -> [String: SharedKVState] {
        [
            "full_attention": SharedKVState(
                keys: MLXArray.ones([1, 1, Swift.max(mainCache.offset, 1), 1]),
                values: MLXArray.ones([1, 1, Swift.max(mainCache.offset, 1), 1]),
                offset: mainCache.offset
            )
        ]
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

final class ScriptedSharedKVDraftModel: Module, SharedKVSpeculativeDraftModel {
    private(set) var draftStepCount = 0

    var speculativeDraftBlockSize: Int {
        2
    }

    deinit {
        // Required by the strict test lint profile.
    }

    func sharedKVDraftOutput(
        tokenEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: SharedKVState],
        position: Int
    ) -> SharedKVDraftOutput {
        draftStepCount += 1
        return SharedKVDraftOutput(
            logits: Self.logits(favoring: [2]),
            hiddenStates: MLXArray.ones(hiddenStates.shape)
        )
    }

    func sharedKVGreedyDraftOutput(
        tokenEmbeddings: MLXArray,
        hiddenStates: MLXArray,
        sharedKVStates: [String: SharedKVState],
        position: Int
    ) -> SharedKVGreedyDraftOutput? {
        draftStepCount += 1
        return SharedKVGreedyDraftOutput(
            token: MLXArray([Int32(2)]),
            hiddenStates: MLXArray.ones(hiddenStates.shape)
        )
    }

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? // swiftlint:disable:this discouraged_optional_collection
    ) -> MLXArray {
        Self.logits(favoring: [2])
    }

    func newCache(parameters: GenerateParameters?) -> [KVCache] {
        []
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
// swiftlint:enable one_declaration_per_file
