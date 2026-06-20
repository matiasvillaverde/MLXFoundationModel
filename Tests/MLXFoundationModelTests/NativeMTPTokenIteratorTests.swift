import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite(
    "Native MTP token iterator",
    .disabled(
        if: ProcessInfo.processInfo.environment["MLX_RUN_ARRAY_UNIT_TESTS"] != "1",
        "Set MLX_RUN_ARRAY_UNIT_TESTS=1 on a launcher with MLX Metal resources available"
    )
)
struct NativeMTPTokenIteratorTests {
    @Test("accepts matching native MTP draft token")
    func acceptsMatchingNativeMTPDraftToken() throws {
        let model = ScriptedNativeMTPModel(verification: .accept)
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(10)])),
            model: model,
            parameters: Self.parameters,
            numDraftTokens: 1
        )

        #expect(iterator.next() == 1)
        #expect(iterator.next() == 2)
        #expect(iterator.next() == 3)
        #expect(model.mainCache.offset == 3)
        #expect(model.mainStepCount == 1)
        #expect(model.verifyStepCount == 1)
        #expect(model.draftStepCount == 1)
    }

    @Test("trims rejected native MTP draft token")
    func trimsRejectedNativeMTPDraftToken() throws {
        let model = ScriptedNativeMTPModel(verification: .reject)
        var iterator = try NativeMTPTokenIterator(
            input: LMInput(tokens: MLXArray([Int32(10)])),
            model: model,
            parameters: Self.parameters,
            numDraftTokens: 1
        )

        #expect(iterator.next() == 1)
        #expect(iterator.next() == 4)
        #expect(model.mainCache.offset == 2)
        #expect(model.mainCache.trimmedTokenCount == 1)
        #expect(model.draftStepCount == 1)
    }

    private static var parameters: GenerateParameters {
        GenerateParameters(maxTokens: 3, temperature: 0)
    }
}
