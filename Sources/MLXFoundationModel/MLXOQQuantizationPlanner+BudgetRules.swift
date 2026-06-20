import Foundation

extension MLXOQQuantizationPlanner {
    func applyMandatoryBoosts(
        state: inout MLXOQQuantizationPlanState,
        hardCap: Double?
    ) {
        for tensor in state.tensors.values {
            guard let spec = mandatorySpec(for: tensor) else {
                continue
            }
            _ = state.applyBoost(spec, to: tensor, hardCapBitsPerWeight: hardCap)
        }
    }

    func applyFractionalExpertBoosts(
        state: inout MLXOQQuantizationPlanState
    ) {
        guard let boost = level.routedExpertDownProjectionBoost else {
            return
        }
        for tensor in state.tensors.values where isRoutedExpertDownProjection(tensor) {
            let spec = MLXOQQuantizationSpec(
                bits: level.baseBits + boost,
                groupSize: defaultGroupSize
            )
            _ = state.applyBoost(spec, to: tensor, hardCapBitsPerWeight: nil, ignoreHardCap: true)
        }
    }

    func applyProtectionFloors(
        state: inout MLXOQQuantizationPlanState,
        hardCap: Double?
    ) {
        for tensor in state.tensors.values {
            guard !isRoutedExpert(tensor),
                let spec = decision(for: tensor).quantizationSpec else {
                continue
            }
            _ = state.applyBoost(spec, to: tensor, hardCapBitsPerWeight: hardCap)
        }
    }

    func applySensitivityBoosts(
        state: inout MLXOQQuantizationPlanState,
        hardCap: Double?,
        layerSensitivityScores: [Int: Double]
    ) {
        let maxScore = layerSensitivityScores.values.max() ?? 0
        for candidate in sensitivityCandidates(state: state, scores: layerSensitivityScores) {
            let maxTargetBits = maxSensitivityTargetBits(
                score: candidate.score,
                maxScore: maxScore,
                currentBits: candidate.spec.bits
            )
            applyBestCandidateBits(
                to: candidate.tensor,
                state: &state,
                currentBits: candidate.spec.bits,
                maxTargetBits: maxTargetBits,
                hardCap: hardCap
            )
        }
    }

    func applyFallbackBoosts(
        state: inout MLXOQQuantizationPlanState,
        target: Double?,
        hardCap: Double?,
        layerSensitivityScores: [Int: Double]
    ) {
        guard let target,
            state.effectiveBitsPerWeight < target else {
            return
        }
        let candidates = boostedCandidates(state: state, scores: layerSensitivityScores)
        for candidate in candidates {
            applyBestCandidateBits(
                to: candidate.tensor,
                state: &state,
                currentBits: candidate.spec.bits,
                maxTargetBits: 8,
                hardCap: hardCap
            )
            if state.effectiveBitsPerWeight >= target {
                return
            }
        }
    }

    private func mandatorySpec(for tensor: MLXOQTensorDescriptor) -> MLXOQQuantizationSpec? {
        let name = tensor.name.lowercased()
        guard containsAny(name, ["lm_head", "embeddings", "embed_tokens", "wte"]) else {
            return nil
        }
        return MLXOQQuantizationSpec(bits: 8, groupSize: defaultGroupSize)
    }

    private func sensitivityCandidates(
        state: MLXOQQuantizationPlanState,
        scores: [Int: Double]
    ) -> [MLXOQQuantizationPlanCandidate] {
        state.tensors.values.compactMap { tensor in
            guard !isRoutedExpert(tensor),
                let layerIndex = ruleContext(for: tensor).layerIndex,
                let spec = state.currentSpec(for: tensor) else {
                return nil
            }
            return MLXOQQuantizationPlanCandidate(
                score: scores[layerIndex] ?? 0,
                spec: spec,
                tensor: tensor
            )
        }
        .sorted { left, right in left.score > right.score }
    }

    private func boostedCandidates(
        state: MLXOQQuantizationPlanState,
        scores: [Int: Double]
    ) -> [MLXOQQuantizationPlanCandidate] {
        state.boosts.compactMap { name, spec in
            guard let tensor = state.tensors[name],
                !isRoutedExpert(tensor) else {
                return nil
            }
            let layerIndex = ruleContext(for: tensor).layerIndex
            return MLXOQQuantizationPlanCandidate(
                score: layerIndex.map { scores[$0] ?? 0 } ?? 0,
                spec: spec,
                tensor: tensor
            )
        }
        .sorted { left, right in left.score > right.score }
    }

    private func maxSensitivityTargetBits(
        score: Double,
        maxScore: Double,
        currentBits: Int
    ) -> Int {
        guard maxScore > 0 else {
            return min(currentBits + 1, 8)
        }
        let ratio = score / maxScore
        if ratio >= 0.5 {
            return 8
        }
        if ratio >= 0.2 {
            return min(currentBits + 2, 8)
        }
        return min(currentBits + 1, 8)
    }

    private func applyBestCandidateBits(
        to tensor: MLXOQTensorDescriptor,
        state: inout MLXOQQuantizationPlanState,
        currentBits: Int,
        maxTargetBits: Int,
        hardCap: Double?
    ) {
        for bits in Self.validBoostBits.reversed() where bits > currentBits && bits <= maxTargetBits {
            let spec = MLXOQQuantizationSpec(bits: bits, groupSize: defaultGroupSize)
            if state.applyBoost(spec, to: tensor, hardCapBitsPerWeight: hardCap) {
                return
            }
        }
    }

    private func isRoutedExpertDownProjection(_ tensor: MLXOQTensorDescriptor) -> Bool {
        let name = tensor.name.lowercased()
        return isRoutedExpert(tensor) && containsAny(name, ["down_proj", "w2"])
    }

    private func isRoutedExpert(_ tensor: MLXOQTensorDescriptor) -> Bool {
        ruleContext(for: tensor).isRoutedExpert
    }

    private func ruleContext(
        for tensor: MLXOQTensorDescriptor
    ) -> MLXOQQuantizationRuleContext {
        MLXOQQuantizationRuleContext(
            tensor: tensor,
            level: level,
            traits: traits,
            defaultGroupSize: defaultGroupSize
        )
    }

    private func containsAny(_ text: String, _ needles: [String]) -> Bool {
        needles.contains { text.contains($0) }
    }

    private static let validBoostBits = [2, 3, 4, 5, 6, 8]
}
