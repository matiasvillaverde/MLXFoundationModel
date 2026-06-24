import MLX
@testable import MLXLocalModels
import Testing

@Suite("Su-scaled RoPE")
struct SuScaledRoPETests {
    @Test("plans short and long LongRoPE frequency tables")
    func plansShortAndLongFrequencies() {
        let plan = SuScaledRoPEPlan(
            dimensions: 4,
            base: 100,
            maxPositionEmbeddings: 128,
            originalMaxPositionEmbeddings: 64,
            shortFactor: [1, 10],
            longFactor: [2, 20]
        )

        #expect(plan.usesLongFrequencies(positionLimit: 64) == false)
        #expect(plan.usesLongFrequencies(positionLimit: 65) == true)
        #expect(plan.frequencyValues(useLongFrequencies: false) == [1, 100])
        #expect(plan.frequencyValues(useLongFrequencies: true) == [2, 200])
        #expect(abs(plan.longScale - 1.0801234) < 0.0001)
    }

    @Test("expands scalar factors and accepts explicit scales")
    func expandsScalarFactorsAndScales() {
        let plan = SuScaledRoPEPlan(
            dimensions: 6,
            maxPositionEmbeddings: 128,
            originalMaxPositionEmbeddings: 64,
            shortFactor: [3],
            longFactor: [5],
            shortMScale: 0.75,
            longMScale: 1.25
        )

        #expect(plan.shortFactor == [3, 3, 3])
        #expect(plan.longFactor == [5, 5, 5])
        #expect(plan.scale(positionLimit: 64) == 0.75)
        #expect(plan.scale(positionLimit: 65) == 1.25)
    }

    @Test("applies scalar offset while preserving non-rotary tail")
    func appliesScalarOffsetWhilePreservingTail() {
        Device.withDefaultDevice(.cpu) {
            let rope = SuScaledRoPE(
                dimensions: 2,
                base: 100,
                maxPositionEmbeddings: 64,
                originalMaxPositionEmbeddings: 64,
                shortFactor: [1],
                longFactor: [10]
            )
            let input = MLXArray([Float(1), 2, 3, 4]).reshaped([1, 1, 1, 4])

            let output = rope(input, offset: 0)

            #expect(output.shape == [1, 1, 1, 4])
            #expect(output.asArray(Float.self) == [1, 2, 3, 4])
        }
    }

    @Test("accepts batch offsets")
    func acceptsBatchOffsets() {
        Device.withDefaultDevice(.cpu) {
            let rope = SuScaledRoPE(
                dimensions: 2,
                maxPositionEmbeddings: 128,
                originalMaxPositionEmbeddings: 64,
                shortFactor: [1],
                longFactor: [1]
            )
            let input = MLXArray.zeros([2, 1, 1, 2])

            let output = rope(input, offset: MLXArray([Int32(0), Int32(70)]))

            #expect(output.shape == [2, 1, 1, 2])
        }
    }
}
