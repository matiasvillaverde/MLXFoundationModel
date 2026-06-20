@testable import MLXLocalModels
import Testing

@Suite("Grammar token masks")
struct GrammarTokenMaskTests {
    @Test("all-allowed states use an empty reject list")
    func allAllowedStatesUseEmptyRejectList() {
        let mask = GrammarTokenMask(bitmask: [Int32(bitPattern: UInt32.max)], vocabularySize: 4)

        #expect(mask.mode == .reject)
        #expect(mask.tokenIDs.isEmpty)
    }

    @Test("sparse allowed ids expand from packed bitsets")
    func sparseAllowedIDsExpandFromPackedBitsets() {
        let mask = GrammarTokenMask(
            bitmask: [
                Int32(bitPattern: 0b1010),
                Int32(bitPattern: UInt32.max)
            ],
            vocabularySize: 34
        )

        #expect(mask.mode == .allow)
        #expect(mask.tokenIDs == [1, 3, 32, 33])
    }

    @Test("padded vocabulary bits are ignored")
    func paddedVocabularyBitsAreIgnored() {
        let mask = GrammarTokenMask(
            bitmask: [
                Int32(bitPattern: UInt32.max),
                Int32(bitPattern: UInt32.max)
            ],
            vocabularySize: 33
        )

        #expect(mask.mode == .reject)
        #expect(mask.tokenIDs.isEmpty)
    }
}
