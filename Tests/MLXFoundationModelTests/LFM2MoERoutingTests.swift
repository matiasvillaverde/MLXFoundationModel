import Foundation
import MLX
@testable import MLXLocalModels
import Testing

@Suite("LFM2 MoE routing")
struct LFM2MoERoutingTests {
    @Test("uses sigmoid scores and keeps expert bias selection-only")
    func usesSigmoidScoresAndKeepsExpertBiasSelectionOnly() {
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)
        let expertBias = MLXArray([Float(0), Float(10), Float(0)])

        let routed = lfm2MoERouter(
            logits: logits,
            expertBias: expertBias,
            topK: 1,
            normTopKProb: false,
            useExpertBias: true,
            routedScalingFactor: 2
        )

        eval(routed.indices, routed.scores)

        #expect(routed.indices.asArray(Int32.self).map(Int.init) == [1])
        #expect(abs(routed.scores.item(Float.self) - (sigmoid(1) * 2)) < 0.0001)
    }

    @Test("normalizes selected sigmoid scores before routed scaling")
    func normalizesSelectedSigmoidScoresBeforeRoutedScaling() {
        let logits = MLXArray([Float(4), Float(1), Float(3)]).reshaped(1, 1, 3)

        let routed = lfm2MoERouter(
            logits: logits,
            expertBias: nil,
            topK: 2,
            normTopKProb: true,
            useExpertBias: false,
            routedScalingFactor: 1.5
        )

        eval(routed.scores)
        let scores = routed.scores.asArray(Float.self)

        #expect(abs(scores.reduce(0, +) - 1.5) < 0.0001)
    }

    private func sigmoid(_ value: Float) -> Float {
        1 / (1 + exp(-value))
    }
}
