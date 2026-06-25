import Foundation
import MLX
import MLXNN

// MARK: - Layout

struct GatedDeltaLayout: Equatable, Sendable {
    let batchSize: Int
    let sequenceLength: Int
    let keyHeads: Int
    let keyDimensions: Int
    let valueHeads: Int
    let valueDimensions: Int

    init(q: MLXArray, k: MLXArray, v: MLXArray) {
        precondition(q.ndim == 4, "gated delta q must be [batch, time, keyHeads, keyDim]")
        precondition(k.ndim == 4, "gated delta k must be [batch, time, keyHeads, keyDim]")
        precondition(v.ndim == 4, "gated delta v must be [batch, time, valueHeads, valueDim]")
        precondition(q.dim(0) == k.dim(0) && q.dim(0) == v.dim(0), "batch sizes differ")
        precondition(q.dim(1) == k.dim(1) && q.dim(1) == v.dim(1), "sequence lengths differ")
        precondition(q.dim(2) == k.dim(2), "q and k head counts differ")
        precondition(q.dim(3) == k.dim(3), "q and k head dimensions differ")
        precondition(v.dim(2).isMultiple(of: q.dim(2)), "value heads must group key heads")

        batchSize = q.dim(0)
        sequenceLength = q.dim(1)
        keyHeads = q.dim(2)
        keyDimensions = q.dim(3)
        valueHeads = v.dim(2)
        valueDimensions = v.dim(3)
    }

    var valueHeadsPerKeyHead: Int { valueHeads / keyHeads }

    var stateShape: [Int] {
        [batchSize, valueHeads, valueDimensions, keyDimensions]
    }

    var supportsMetalKernel: Bool {
        keyDimensions > 0 && keyDimensions.isMultiple(of: 32)
    }
}

// MARK: - Decay

enum GatedDeltaDecay {
    static func values(aLog: MLXArray, a: MLXArray, dtBias: MLXArray) -> MLXArray {
        let rate = exp(aLog.asType(.float32))
        let step = softplus(a + dtBias)
        return exp(-rate * step).asType(a.dtype)
    }
}

func computeGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray) -> MLXArray {
    GatedDeltaDecay.values(aLog: aLog, a: a, dtBias: dtBias)
}

// MARK: - Metal Kernel

private func makeGatedDeltaKernel(hasMask: Bool) -> MLXFast.MLXFastKernel? {
    let isTokenActive = hasMask ? "mask[batchIndex * sequenceLength + tokenIndex]" : "true"
    let suffix = hasMask ? "_masked" : ""

    let source = """
            auto sampleAndValueHead = thread_position_in_grid.z;
            auto batchIndex = sampleAndValueHead / valueHeadCount;
            auto valueHead = sampleAndValueHead % valueHeadCount;
            auto keyHead = valueHead / valueHeadsPerKeyHead;

            constexpr int keyPiecesPerThread = keyDimensions / 32;

            auto qToken = q + batchIndex * sequenceLength * keyHeadCount * keyDimensions
                + keyHead * keyDimensions;
            auto kToken = k + batchIndex * sequenceLength * keyHeadCount * keyDimensions
                + keyHead * keyDimensions;
            auto vToken = v + batchIndex * sequenceLength * valueHeadCount * valueDimensions
                + valueHead * valueDimensions;
            auto yToken = y + batchIndex * sequenceLength * valueHeadCount * valueDimensions
                + valueHead * valueDimensions;

            auto decayToken = decay + batchIndex * sequenceLength * valueHeadCount;
            auto betaToken = beta + batchIndex * sequenceLength * valueHeadCount;

            auto keyLane = thread_position_in_threadgroup.x;
            auto valueChannel = thread_position_in_grid.y;

            auto stateOffset = (sampleAndValueHead * valueDimensions + valueChannel)
                * keyDimensions;
            auto stateRead = stateIn + stateOffset;
            auto stateWrite = stateOut + stateOffset;

            float laneState[keyPiecesPerThread];
            for (int piece = 0; piece < keyPiecesPerThread; ++piece) {
                auto keyOffset = keyPiecesPerThread * keyLane + piece;
                laneState[piece] = static_cast<float>(stateRead[keyOffset]);
            }

            for (int tokenIndex = 0; tokenIndex < sequenceLength; ++tokenIndex) {
                if (\(isTokenActive)) {
                    float memory = 0.0f;
                    for (int piece = 0; piece < keyPiecesPerThread; ++piece) {
                        auto keyOffset = keyPiecesPerThread * keyLane + piece;
                        laneState[piece] *= decayToken[valueHead];
                        memory += laneState[piece] * kToken[keyOffset];
                    }
                    memory = simd_sum(memory);

                    auto correction = (vToken[valueChannel] - memory) * betaToken[valueHead];

                    float output = 0.0f;
                    for (int piece = 0; piece < keyPiecesPerThread; ++piece) {
                        auto keyOffset = keyPiecesPerThread * keyLane + piece;
                        laneState[piece] += kToken[keyOffset] * correction;
                        output += laneState[piece] * qToken[keyOffset];
                    }
                    output = simd_sum(output);

                    if (thread_index_in_simdgroup == 0) {
                        yToken[valueChannel] = static_cast<InT>(output);
                    }
                } else {
                    if (thread_index_in_simdgroup == 0) {
                        yToken[valueChannel] = static_cast<InT>(0.0f);
                    }
                }

                qToken += keyHeadCount * keyDimensions;
                kToken += keyHeadCount * keyDimensions;
                vToken += valueHeadCount * valueDimensions;
                yToken += valueHeadCount * valueDimensions;
                decayToken += valueHeadCount;
                betaToken += valueHeadCount;
            }

            for (int piece = 0; piece < keyPiecesPerThread; ++piece) {
                auto keyOffset = keyPiecesPerThread * keyLane + piece;
                stateWrite[keyOffset] = static_cast<InT>(laneState[piece]);
            }
        """

    var inputNames = ["q", "k", "v", "decay", "beta", "stateIn", "sequenceLength"]
    if hasMask {
        inputNames.append("mask")
    }

    return MLXFast.metalKernel(
        name: "gated_delta_recurrence\(suffix)",
        inputNames: inputNames,
        outputNames: ["y", "stateOut"],
        source: source
    )
}

private final class GatedDeltaKernelPool: Sendable {
    static let shared = GatedDeltaKernelPool()

    let unmasked: MLXFast.MLXFastKernel?
    let masked: MLXFast.MLXFastKernel?

    private init() {
        unmasked = makeGatedDeltaKernel(hasMask: false)
        masked = makeGatedDeltaKernel(hasMask: true)
    }
}

// MARK: - Kernel Dispatch

func gatedDeltaKernel(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let layout = GatedDeltaLayout(q: q, k: k, v: v)
    let kernel = mask == nil ? GatedDeltaKernelPool.shared.unmasked : GatedDeltaKernelPool.shared.masked

    guard layout.supportsMetalKernel, let kernel else {
        return gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
    }

    var inputs: [MLXArray] = [q, k, v, g, beta, state, MLXArray(layout.sequenceLength)]
    if let mask {
        inputs.append(mask)
    }

    let outputs = kernel(
        inputs,
        template: [
            ("InT", q.dtype),
            ("keyDimensions", layout.keyDimensions),
            ("valueDimensions", layout.valueDimensions),
            ("keyHeadCount", layout.keyHeads),
            ("valueHeadCount", layout.valueHeads),
            ("valueHeadsPerKeyHead", layout.valueHeadsPerKeyHead),
        ],
        grid: (32, layout.valueDimensions, layout.batchSize * layout.valueHeads),
        threadGroup: (32, 4, 1),
        outputShapes: [[
            layout.batchSize,
            layout.sequenceLength,
            layout.valueHeads,
            layout.valueDimensions,
        ], state.shape],
        outputDTypes: [q.dtype, q.dtype]
    )

    return (outputs[0], outputs[1])
}

// MARK: - Ops Fallback

private func expandedStepMask(_ mask: MLXArray) -> MLXArray {
    switch mask.ndim {
    case 0:
        mask
    case 1:
        expandedDimensions(mask, axes: [1, 2, 3])
    case 2:
        expandedDimensions(mask, axes: [2, 3])
    case 3:
        expandedDimensions(mask, axis: -1)
    case 4:
        mask
    default:
        fatalError("Unsupported gated delta mask shape \(mask.shape)")
    }
}

private func expandedOutputMask(_ mask: MLXArray) -> MLXArray {
    switch mask.ndim {
    case 0:
        mask
    case 1:
        expandedDimensions(mask, axes: [1, 2])
    case 2:
        expandedDimensions(mask, axis: -1)
    case 3:
        mask
    default:
        fatalError("Unsupported gated delta output mask shape \(mask.shape)")
    }
}

private func maskSlice(_ mask: MLXArray?, at tokenIndex: Int) -> MLXArray? {
    guard let mask else { return nil }
    switch mask.ndim {
    case 1:
        return mask[tokenIndex]
    case 2:
        return mask[0..., tokenIndex]
    case 3:
        return mask[0..., tokenIndex, 0...]
    case 4:
        return mask[0..., tokenIndex, 0..., 0...]
    default:
        fatalError("Unsupported gated delta mask shape \(mask.shape)")
    }
}

private func gatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let previousState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gated delta decay shape \(g.shape)")
    }

    var nextState = state * decay
    let memory = (nextState * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let correction = (v - memory) * expandedDimensions(beta, axis: -1)
    nextState = nextState + expandedDimensions(k, axis: -2) * expandedDimensions(correction, axis: -1)
    var output = (nextState * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        nextState = MLX.where(expandedStepMask(mask), nextState, previousState)
        output = MLX.where(expandedOutputMask(mask), output, MLXArray.zeros(like: output))
    }

    return (output, nextState)
}

func gatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let layout = GatedDeltaLayout(q: q, k: k, v: v)

    var queries = q
    var keys = k
    if layout.valueHeadsPerKeyHead > 1 {
        queries = repeated(queries, count: layout.valueHeadsPerKeyHead, axis: -2)
        keys = repeated(keys, count: layout.valueHeadsPerKeyHead, axis: -2)
    }

    var state = state ?? MLXArray.zeros(layout.stateShape, dtype: q.dtype)
    var outputs = [MLXArray]()
    outputs.reserveCapacity(layout.sequenceLength)

    for tokenIndex in 0 ..< layout.sequenceLength {
        let (output, nextState) = gatedDeltaStepOps(
            q: queries[0..., tokenIndex],
            k: keys[0..., tokenIndex],
            v: v[0..., tokenIndex],
            g: g[0..., tokenIndex],
            beta: beta[0..., tokenIndex],
            state: state,
            mask: maskSlice(mask, at: tokenIndex)
        )
        outputs.append(output)
        state = nextState
    }

    return (MLX.stacked(outputs, axis: 1), state)
}

// MARK: - Public API

func gatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil
) -> (MLXArray, MLXArray) {
    let layout = GatedDeltaLayout(q: q, k: k, v: v)
    let beta = sigmoid(b)
    let decay = computeGatedDeltaG(aLog, a, dtBias)
    let initialState = state ?? MLXArray.zeros(layout.stateShape, dtype: q.dtype)

    guard layout.supportsMetalKernel else {
        return gatedDeltaOps(q: q, k: k, v: v, g: decay, beta: beta, state: initialState, mask: mask)
    }

    return gatedDeltaKernel(
        q: q,
        k: k,
        v: v,
        g: decay,
        beta: beta,
        state: initialState,
        mask: mask
    )
}
