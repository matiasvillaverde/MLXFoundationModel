import MLX
@testable import MLXLocalModels
import MLXNN
import Testing

@Suite("Switch expert layers")
struct SwitchLayersTests {
    @Test("sorts and restores expert assignments")
    func sortsAndRestoresExpertAssignments() {
        Device.withDefaultDevice(.cpu) {
            let input = MLXArray((0 ..< 8).map(Float.init)).reshaped([4, 1, 1, 2])
            let indices = MLXArray([Int32(2), 0, 1, 2, 0, 1, 2, 1]).reshaped([4, 2])

            let permutation = SwitchExpertPermutation(input: input, expertIndices: indices)
            let restored = permutation.restore(permutation.sortedInput)

            #expect(permutation.sortedExpertIndices.asArray(Int32.self) == [0, 0, 1, 1, 1, 2, 2, 2])
            #expect(restored.shape == [4, 2, 1, 2])
            #expect(
                restored.asArray(Float.self) == [
                    0, 1, 0, 1,
                    2, 3, 2, 3,
                    4, 5, 4, 5,
                    6, 7, 6, 7
                ]
            )
        }
    }

    @Test("switch linear gathers deterministic expert projections")
    func switchLinearGathersDeterministicExpertProjections() throws {
        try Device.withDefaultDevice(.cpu) {
            let layer = SwitchLinear(inputDims: 2, outputDims: 2, numExperts: 3, bias: true)
            let weight = MLXArray([
                Float(1), 0, 0, 1,
                2, 0, 0, 2,
                1, 1, 1, -1
            ]).reshaped([3, 2, 2])
            let bias = MLXArray([
                Float(0), 0,
                1, 1,
                -1, 1
            ]).reshaped([3, 2])
            try layer.update(
                parameters: ModuleParameters.unflattened([
                    "weight": weight,
                    "bias": bias
                ]),
                verify: .all
            )

            let input = MLXArray([Float(1), 2, 3, 4, 5, 6]).reshaped([3, 1, 1, 2])
            let indices = MLXArray([Int32(0), 1, 2]).reshaped([3, 1])

            let output = layer(input, indices)

            #expect(output.shape == [3, 1, 1, 2])
            #expect(output.asArray(Float.self) == [1, 2, 7, 9, 10, 0])
        }
    }

    @Test("switch GLU matches manual routing for small and sorted dispatch")
    func switchGLUMatchesManualRouting() throws {
        try Device.withDefaultDevice(.cpu) {
            try Self.expectSwitchGLUMatchesManualRouting(tokenCount: 4)
            try Self.expectSwitchGLUMatchesManualRouting(
                tokenCount: SwitchExpertDispatch.sortedDispatchThreshold + 1
            )
        }
    }

    private static func expectSwitchGLUMatchesManualRouting(tokenCount: Int) throws {
        let layer = SwitchGLU(
            inputDims: 1,
            hiddenDims: 1,
            numExperts: 2,
            activation: { $0 },
            bias: false
        )
        try layer.update(
            parameters: ModuleParameters.unflattened([
                "up_proj.weight": MLXArray([Float(2), 3]).reshaped([2, 1, 1]),
                "gate_proj.weight": MLXArray([Float(5), 7]).reshaped([2, 1, 1]),
                "down_proj.weight": MLXArray([Float(11), 13]).reshaped([2, 1, 1])
            ]),
            verify: .all
        )

        let input = MLXArray((1 ... tokenCount).map(Float.init)).reshaped([1, tokenCount, 1])
        let expertIndices = (0 ..< tokenCount).map { Int32($0 % 2) }
        let indices = MLXArray(expertIndices).reshaped([1, tokenCount, 1])

        let output = layer(input, indices).asArray(Float.self)
        let expected = (0 ..< tokenCount).map { tokenIndex -> Float in
            let value = Float(tokenIndex + 1)
            let coefficient: Float = expertIndices[tokenIndex] == 0 ? 2 * 5 * 11 : 3 * 7 * 13
            return value * value * coefficient
        }

        #expect(output.count == expected.count)
        for (actual, expectedValue) in zip(output, expected) {
            #expect(abs(actual - expectedValue) < 0.001)
        }
    }
}
