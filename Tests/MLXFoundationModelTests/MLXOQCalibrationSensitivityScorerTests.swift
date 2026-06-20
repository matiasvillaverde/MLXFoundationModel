import Foundation
import MLX
@testable import MLXFoundationModel
import Testing

@Suite("MLX oQ calibration sensitivity scorer")
struct MLXOQCalibrationSensitivityScorerTests {
    @Test("relative MSE scores aggregate into budget planning")
    func relativeMSEScoresAggregateIntoBudgetPlanning() throws {
        let scorer = MLXOQCalibrationSensitivityScorer()
        let sensitive = try Self.observation(
            name: "model.layers.4.self_attn.q_proj.weight",
            candidate: [1, 1, 1, 1]
        )
        let ordinary = try Self.observation(
            name: "model.layers.5.self_attn.q_proj.weight",
            candidate: [1, 2, 2.9, 4]
        )

        let scores = try scorer.layerSensitivityScores(for: [sensitive, ordinary])
        let plan = try Self.plan(scores: scores)

        #expect(scores[4] ?? 0 > scores[5] ?? 0)
        #expect(plan.boosts["model.layers.4.self_attn.q_proj.weight"]?.bits == 6)
        #expect(plan.boosts["model.layers.5.self_attn.q_proj.weight"] == nil)
    }

    @Test("projection scoring quantizes weights and compares real MLX outputs")
    func projectionScoringQuantizesWeightsAndComparesRealMLXOutputs() throws {
        try Device.withDefaultDevice(.cpu) {
            let scorer = MLXOQCalibrationSensitivityScorer()
            let score = try scorer.scoreProjection(
                tensorName: "model.layers.3.mlp.gate_proj.weight",
                inputs: Self.inputs(),
                weight: Self.weight(),
                spec: .init(bits: 2, groupSize: 64)
            )

            #expect(score.layerIndex == 3)
            #expect(score.tensorName == "model.layers.3.mlp.gate_proj.weight")
            #expect(score.score.isFinite)
            #expect(score.score > 0)
        }
    }

    @Test("plan options accept typed calibration scores")
    func planOptionsAcceptTypedCalibrationScores() throws {
        let tensor = Self.tensor("model.layers.4.self_attn.q_proj.weight")
        let planner = try Self.planner()
        let options = MLXOQQuantizationPlanOptions(
            calibrationScores: [
                .init(layerIndex: 4, score: 2, tensorName: tensor.name)
            ],
            targetBitsPerWeight: 8,
            hardCapBitsPerWeight: 8
        )

        let plan = planner.plan(for: [tensor], options: options)

        #expect(plan.boosts[tensor.name]?.bits == 6)
    }

    private static func observation(
        name: String,
        candidate: [Float]
    ) throws -> MLXOQCalibrationObservation {
        try MLXOQCalibrationObservation(
            tensorName: name,
            referenceOutput: MLXArray([Float(1), 2, 3, 4]),
            candidateOutput: MLXArray(candidate)
        )
    }

    private static func plan(scores: [Int: Double]) throws -> MLXOQQuantizationPlan {
        let planner = try Self.planner()
        return planner.plan(
            for: [
                Self.tensor("model.layers.4.self_attn.q_proj.weight"),
                Self.tensor("model.layers.5.self_attn.q_proj.weight")
            ],
            options: .init(
                targetBitsPerWeight: 5.6,
                hardCapBitsPerWeight: 5.6,
                layerSensitivityScores: scores
            )
        )
    }

    private static func planner() throws -> MLXOQQuantizationPlanner {
        try #require(MLXOQQuantizationPlanner(level: "oQ4", traits: .init(numLayers: 32)))
    }

    private static func tensor(_ name: String) -> MLXOQTensorDescriptor {
        MLXOQTensorDescriptor(name: name, shape: [4_096, 4_096])
    }

    private static func inputs() -> MLXArray {
        MLXArray((0..<128).map { Float($0 % 11) / 7 }).reshaped([2, 64])
    }

    private static func weight() -> MLXArray {
        MLXArray((0..<128).map { Float(($0 % 17) - 8) / 5 }).reshaped([2, 64])
    }
}
