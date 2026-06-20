@testable import MLXFoundationModel
import Testing

@Suite("MLX oQ quantization planner")
struct MLXOQQuantizationPlannerTests {
    @Test("parses supported oQ levels")
    func parsesSupportedLevels() throws {
        let level = try #require(MLXOQLevel("DeepSeek-V4-oQ2.7-fp16"))

        #expect(level.value == "2.7")
        #expect(level.label == "oQ2.7")
        #expect(level.baseBits == 2)
        #expect(level.routedExpertDownProjectionBoost == 2)
        #expect(MLXOQLevel("oQ7") == nil)
    }

    @Test("mirrors oMLX source quantizable model filtering")
    func mirrorsSourceQuantizableModelFiltering() {
        #expect(MLXOQQuantizationPlanner.isQuantizableModel(config: [:]))
        #expect(!MLXOQQuantizationPlanner.isQuantizableModel(config: [
            "quantization": ["bits": 4]
        ]))
        #expect(!MLXOQQuantizationPlanner.isQuantizableModel(config: [
            "quantization_config": ["quant_method": "affine"]
        ]))
        #expect(MLXOQQuantizationPlanner.isQuantizableModel(config: [
            "quantization_config": ["quant_method": "fp8"]
        ]))
    }

    @Test("keeps protected tensors in full precision")
    func keepsProtectedTensorsFullPrecision() throws {
        let planner = try Self.planner(level: "oQ4")

        #expect(planner.decision(for: Self.tensor("model.layers.0.mlp.gate.weight")) ==
            .keepFullPrecision)
        #expect(planner.decision(for: Self.tensor("visual.patch_embed.proj.weight")) ==
            .keepFullPrecision)
        #expect(planner.decision(for: Self.tensor("model.layers.0.input_layernorm.weight")) ==
            .keepFullPrecision)
        #expect(planner.decision(for: Self.tensor("model.mtp.fc.weight")) == .keepFullPrecision)
    }

    @Test("applies fixed high precision rules")
    func appliesFixedHighPrecisionRules() throws {
        let planner = try Self.planner(level: "oQ4")

        #expect(Self.specBits(planner, "lm_head.weight") == 6)
        #expect(Self.specBits(planner, "model.shared_expert.down_proj.weight") == 8)
        #expect(Self.specBits(planner, "model.layers.4.self_attn.q_a_proj.weight") == 6)
        #expect(Self.specBits(planner, "model.layers.4.linear_attn.conv1d.weight") == 8)
    }

    @Test("applies layer sensitivity projection floors")
    func appliesLayerSensitivityProjectionFloors() throws {
        let planner = try Self.planner(level: "oQ4", traits: .init(numLayers: 32))

        #expect(Self.specBits(planner, "model.layers.0.self_attn.v_proj.weight") == 6)
        #expect(Self.specBits(planner, "model.layers.16.self_attn.v_proj.weight") == 4)
        #expect(Self.specBits(planner, "model.layers.31.self_attn.q_proj.weight") == 5)
        #expect(Self.specBits(planner, "model.layers.16.mlp.down_proj.weight") == 5)
    }

    @Test("applies fractional routed expert down projection boost")
    func appliesFractionalRoutedExpertDownProjectionBoost() throws {
        let traits = MLXOQModelQuantizationTraits(
            numLayers: 32,
            numExperts: 256,
            hiddenSize: 4_096
        )
        let planner = try Self.planner(level: "oQ2.7", traits: traits)

        #expect(Self.specBits(
            planner,
            "model.layers.10.block_sparse_moe.experts.3.down_proj.weight"
        ) == 4)
    }

    @Test("applies large MoE expert protection before fractional boost")
    func appliesLargeMoEExpertProtectionBeforeFractionalBoost() throws {
        let traits = MLXOQModelQuantizationTraits(
            numLayers: 32,
            numExperts: 512,
            hiddenSize: 4_096
        )
        let planner = try Self.planner(level: "oQ2.7", traits: traits)

        #expect(Self.specBits(
            planner,
            "model.layers.10.block_sparse_moe.experts.3.gate_proj.weight"
        ) == 4)
        #expect(Self.specBits(
            planner,
            "model.layers.10.block_sparse_moe.experts.3.down_proj.weight"
        ) == 3)
    }

    @Test("returns base quantization for default quantizable tensors")
    func returnsBaseQuantizationForDefaultTensors() throws {
        let planner = try Self.planner(level: "oQ5")
        let decision = planner.decision(for: Self.tensor("model.layers.8.mlp.gate_proj.weight"))

        #expect(decision.quantizationSpec?.bits == 5)
        #expect(decision.quantizationSpec?.groupSize == 64)
        #expect(decision.quantizationSpec?.mode == "affine")
    }

    @Test("keeps tensors with incompatible group width in full precision")
    func keepsIncompatibleGroupWidthFullPrecision() throws {
        let planner = try Self.planner(level: "oQ4")
        let tensor = MLXOQTensorDescriptor(
            name: "model.layers.8.mlp.gate_proj.weight",
            shape: [4_096, 63]
        )

        #expect(planner.decision(for: tensor) == .keepFullPrecision)
    }

    @Test("budgeted plan applies mandatory boosts under hard cap")
    func budgetedPlanAppliesMandatoryBoostsUnderHardCap() throws {
        let planner = try Self.planner(level: "oQ4")
        let tensors = Self.defaultLayerTensors(count: 19) + [
            Self.tensor("lm_head.weight")
        ]

        let plan = planner.plan(for: tensors)

        #expect(plan.baselineBitsPerWeight == 4.5)
        #expect(plan.boosts["lm_head.weight"]?.bits == 8)
        #expect(plan.effectiveBitsPerWeight <= 4.7)
        #expect(plan.quantizedParameterCount == tensors.count * 4_096 * 4_096)
    }

    @Test("budgeted plan prices fixed overrides and excludes them from boosts")
    func budgetedPlanPricesFixedOverridesAndExcludesThemFromBoosts() throws {
        let planner = try Self.planner(level: "oQ4")
        let fixed = MLXOQQuantizationSpec(bits: 8, groupSize: 64, mode: "mxfp8")
        let tensor = Self.tensor("model.layers.8.self_attn.q_proj.weight")

        let plan = planner.plan(
            for: [tensor],
            options: .init(fixedOverrides: [tensor.name: fixed])
        )

        #expect(plan.fixedOverrides[tensor.name] == fixed)
        #expect(plan.decisions[tensor.name]?.quantizationSpec == fixed)
        #expect(plan.boosts[tensor.name] == nil)
        #expect(plan.effectiveBitsPerWeight == 8.25)
    }

    @Test("budgeted plan applies sensitivity boosts before lower scoring layers")
    func budgetedPlanAppliesSensitivityBoostsBeforeLowerScoringLayers() throws {
        let traits = MLXOQModelQuantizationTraits(numLayers: 32)
        let planner = try Self.planner(level: "oQ4", traits: traits)
        let sensitive = Self.tensor("model.layers.4.self_attn.q_proj.weight")
        let ordinary = Self.tensor("model.layers.5.self_attn.q_proj.weight")

        let plan = planner.plan(
            for: [sensitive, ordinary],
            options: .init(
                targetBitsPerWeight: 5.6,
                hardCapBitsPerWeight: 5.6,
                layerSensitivityScores: [4: 10, 5: 1]
            )
        )

        #expect(plan.boosts[sensitive.name]?.bits == 6)
        #expect(plan.boosts[ordinary.name] == nil)
        #expect(plan.effectiveBitsPerWeight == 5.5)
    }

    @Test("budgeted plan keeps fractional expert boost even when hard cap is tight")
    func budgetedPlanKeepsFractionalExpertBoostWithTightHardCap() throws {
        let traits = MLXOQModelQuantizationTraits(
            numLayers: 32,
            numExperts: 256,
            hiddenSize: 4_096
        )
        let planner = try Self.planner(level: "oQ2.7", traits: traits)
        let expert = Self.tensor("model.layers.10.block_sparse_moe.experts.3.down_proj.weight")

        let plan = planner.plan(
            for: [expert],
            options: .init(hardCapBitsPerWeight: 2.9)
        )

        #expect(plan.boosts[expert.name]?.bits == 4)
        #expect(plan.effectiveBitsPerWeight == 4.5)
    }

    private static func planner(
        level: String,
        traits: MLXOQModelQuantizationTraits = .init()
    ) throws -> MLXOQQuantizationPlanner {
        try #require(MLXOQQuantizationPlanner(level: level, traits: traits))
    }

    private static func tensor(_ name: String) -> MLXOQTensorDescriptor {
        MLXOQTensorDescriptor(name: name, shape: [4_096, 4_096])
    }

    private static func defaultLayerTensors(count: Int) -> [MLXOQTensorDescriptor] {
        (0..<count).map { index in
            tensor("model.layers.\(index).mlp.gate_proj.weight")
        }
    }

    private static func specBits(
        _ planner: MLXOQQuantizationPlanner,
        _ tensorName: String
    ) -> Int? {
        planner.decision(for: tensor(tensorName)).quantizationSpec?.bits
    }
}
