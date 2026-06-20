import MLX

internal struct MLXQuantizedWeightSanitizer {
    private enum DotScalePair: Equatable, Sendable {
        case mxfp4
        case mxfp8(rowRepeat: Int, columnRepeat: Int)
    }

    private struct BlockShape: Equatable, Sendable {
        let rows: Int
        let columns: Int
    }

    internal enum Strategy: Equatable, Sendable {
        case automatic(blockSize: Int = 128)
        case direct
        case block(blockSize: Int = 128)
    }

    internal enum SidecarPolicy: Equatable, Sendable {
        case keepActivationScale
        case dropActivationScale
    }

    internal struct Report: Equatable, Sendable {
        internal var directScaleCount = 0
        internal var blockScaleCount = 0
        internal var missingWeightCount = 0
        internal var incompatibleScaleCount = 0
        internal var packedScaleCount = 0
        internal var droppedActivationScaleCount = 0

        internal var dequantizedCount: Int {
            directScaleCount + blockScaleCount + packedScaleCount
        }
    }

    internal struct Result {
        internal let weights: [String: MLXArray]
        internal let report: Report
    }

    internal static func sanitize(
        _ weights: [String: MLXArray],
        strategy: Strategy = .automatic(),
        sidecarPolicy: SidecarPolicy = .dropActivationScale
    ) -> Result {
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(weights.count)
        var report = Report()

        for (key, value) in weights {
            if key.contains("weight_scale_inv") {
                let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
                guard let weight = weights[weightKey] else {
                    report.missingWeightCount += 1
                    continue
                }
                switch scaledWeight(weight: weight, scaleInv: value, strategy: strategy) {
                case .direct(let scaled):
                    sanitized[weightKey] = scaled
                    report.directScaleCount += 1
                case .block(let scaled):
                    sanitized[weightKey] = scaled
                    report.blockScaleCount += 1
                case .incompatible:
                    sanitized[weightKey] = weight
                    report.incompatibleScaleCount += 1
                }
                continue
            }

            if sidecarPolicy == .dropActivationScale, key.contains("activation_scale") {
                report.droppedActivationScaleCount += 1
                continue
            }

            if sanitized[key] == nil {
                sanitized[key] = value
            }
        }

        return Result(weights: sanitized, report: report)
    }

    internal static func sanitizePackedScalePairs(
        _ weights: [String: MLXArray],
        metadata: [String: String]
    ) -> Result {
        var sanitized = [String: MLXArray]()
        sanitized.reserveCapacity(weights.count)
        var report = Report()

        for (key, value) in weights {
            if let packed = packedScaleWeight(
                scaleKey: key,
                scale: value,
                weights: weights,
                metadata: metadata
            ) {
                sanitized[packed.weightKey] = packed.weight
                report.packedScaleCount += 1
                continue
            }

            if sanitized[key] == nil {
                sanitized[key] = value
            }
        }

        return Result(weights: sanitized, report: report)
    }

    private enum ScaledWeight {
        case direct(MLXArray)
        case block(MLXArray)
        case incompatible
    }

    private static func scaledWeight(
        weight: MLXArray,
        scaleInv: MLXArray,
        strategy: Strategy
    ) -> ScaledWeight {
        switch strategy {
        case .direct:
            return .direct(weight * scaleInv)
        case .block(let blockSize):
            return blockScaledWeight(
                weight: weight,
                scaleInv: scaleInv,
                blockShape: BlockShape(rows: blockSize, columns: blockSize)
            )
        case .automatic(let blockSize):
            if let blockShape = inferredBlockShape(
                weight: weight,
                scaleInv: scaleInv,
                preferredBlockSize: blockSize
            ) {
                return blockScaledWeight(weight: weight, scaleInv: scaleInv, blockShape: blockShape)
            }
            return .direct(weight * scaleInv)
        }
    }

    private static func inferredBlockShape(
        weight: MLXArray,
        scaleInv: MLXArray,
        preferredBlockSize: Int
    ) -> BlockShape? {
        guard weight.ndim == 2, scaleInv.ndim == 2 else {
            return nil
        }
        guard scaleInv.shape != weight.shape else {
            return nil
        }
        guard let rowBlock = inferredBlockSize(
            dimension: weight.dim(0),
            blockCount: scaleInv.dim(0),
            preferredBlockSize: preferredBlockSize
        ),
            let columnBlock = inferredBlockSize(
                dimension: weight.dim(1),
                blockCount: scaleInv.dim(1),
                preferredBlockSize: preferredBlockSize
            )
        else {
            return nil
        }
        return BlockShape(rows: rowBlock, columns: columnBlock)
    }

    private static func inferredBlockSize(
        dimension: Int,
        blockCount: Int,
        preferredBlockSize: Int
    ) -> Int? {
        guard dimension > 0, blockCount > 0 else {
            return nil
        }
        for candidate in blockSizeCandidates(preferredBlockSize: preferredBlockSize) {
            if ceilDiv(dimension, candidate) == blockCount {
                return candidate
            }
        }
        if dimension.isMultiple(of: blockCount) {
            return dimension / blockCount
        }
        return nil
    }

    private static func blockSizeCandidates(preferredBlockSize: Int) -> [Int] {
        var candidates = [preferredBlockSize, 128, 64, 32, 256, 16, 8].filter { $0 > 0 }
        var seen = Set<Int>()
        candidates.removeAll { candidate in
            !seen.insert(candidate).inserted
        }
        return candidates
    }

    private static func ceilDiv(_ numerator: Int, _ denominator: Int) -> Int {
        (numerator + denominator - 1) / denominator
    }

    private static func blockScaledWeight(
        weight: MLXArray,
        scaleInv: MLXArray,
        blockShape: BlockShape
    ) -> ScaledWeight {
        guard weight.ndim == 2, scaleInv.ndim == 2 else {
            return .incompatible
        }

        let dtype = weight.dtype
        let rows = weight.dim(0)
        let columns = weight.dim(1)
        let paddedRows = scaleInv.dim(0) * blockShape.rows
        let paddedColumns = scaleInv.dim(1) * blockShape.columns
        guard paddedRows >= rows, paddedColumns >= columns else {
            return .incompatible
        }

        let paddedWeight = padded(
            weight,
            widths: [
                .init((0, paddedRows - rows)),
                .init((0, paddedColumns - columns))
            ]
        )
        let blocked = paddedWeight.reshaped([
            scaleInv.dim(0),
            blockShape.rows,
            scaleInv.dim(1),
            blockShape.columns
        ])
        let scaled = blocked * scaleInv[0..., .newAxis, 0..., .newAxis]
        return .block(
            scaled
                .reshaped([paddedRows, paddedColumns])[0 ..< rows, 0 ..< columns]
                .asType(dtype)
        )
    }

    private static func packedScaleWeight(
        scaleKey: String,
        scale: MLXArray,
        weights: [String: MLXArray],
        metadata: [String: String]
    ) -> (weightKey: String, weight: MLXArray)? {
        guard let weightKey = dotScaleWeightKey(for: scaleKey),
            let weight = weights[weightKey],
            let pair = dotScalePair(
                weightKey: weightKey,
                weight: weight,
                scaleKey: scaleKey,
                scale: scale,
                metadata: metadata
            )
        else {
            return nil
        }
        return (weightKey, dequantizedPackedWeight(weight: weight, scale: scale, pair: pair))
    }

    private static func dotScaleWeightKey(for scaleKey: String) -> String? {
        guard scaleKey.hasSuffix(".scale") else {
            return nil
        }
        return String(scaleKey.dropLast(".scale".count)) + ".weight"
    }

    private static func dotScalePair(
        weightKey: String,
        weight: MLXArray,
        scaleKey: String,
        scale: MLXArray,
        metadata: [String: String]
    ) -> DotScalePair? {
        let weightDType = tensorDType(weightKey, metadata: metadata)
        let scaleDType = tensorDType(scaleKey, metadata: metadata)
        guard weight.ndim == 2, scale.ndim == 2, scaleDType == "F8_E8M0" else {
            return nil
        }
        if isMXFP4Pair(weight: weight, scale: scale, weightDType: weightDType) {
            return .mxfp4
        }
        return mxfp8Pair(weight: weight, scale: scale, weightDType: weightDType)
    }

    private static func isMXFP4Pair(
        weight: MLXArray,
        scale: MLXArray,
        weightDType: String?
    ) -> Bool {
        weightDType == "I8"
            && weight.dim(1).isMultiple(of: 16)
            && scale.shape == [weight.dim(0), weight.dim(1) / 16]
    }

    private static func mxfp8Pair(
        weight: MLXArray,
        scale: MLXArray,
        weightDType: String?
    ) -> DotScalePair? {
        guard weightDType == "F8_E4M3",
            scale.dim(0) > 0,
            scale.dim(1) > 0,
            weight.dim(0).isMultiple(of: scale.dim(0)),
            weight.dim(1).isMultiple(of: scale.dim(1)),
            (weight.dim(1) / scale.dim(1)).isMultiple(of: 32),
            weight.dim(1).isMultiple(of: 4)
        else {
            return nil
        }
        return .mxfp8(
            rowRepeat: weight.dim(0) / scale.dim(0),
            columnRepeat: (weight.dim(1) / 32) / scale.dim(1)
        )
    }

    private static func dequantizedPackedWeight(
        weight: MLXArray,
        scale: MLXArray,
        pair: DotScalePair
    ) -> MLXArray {
        switch pair {
        case .mxfp4:
            dequantized(
                weight.view(dtype: .uint32),
                scales: scale,
                biases: nil,
                groupSize: 32,
                bits: 4,
                mode: .mxfp4,
                dtype: .bfloat16
            )
        case .mxfp8(let rowRepeat, let columnRepeat):
            dequantized(
                weight.view(dtype: .uint32),
                scales: repeatedScale(scale, rowRepeat: rowRepeat, columnRepeat: columnRepeat),
                biases: nil,
                groupSize: 32,
                bits: 8,
                mode: .mxfp8,
                dtype: .bfloat16
            )
        }
    }

    private static func repeatedScale(
        _ scale: MLXArray,
        rowRepeat: Int,
        columnRepeat: Int
    ) -> MLXArray {
        var result = scale
        if columnRepeat > 1 {
            result = repeated(result, count: columnRepeat, axis: -1)
        }
        if rowRepeat > 1 {
            result = repeated(result, count: rowRepeat, axis: 0)
        }
        return result
    }

    private static func tensorDType(
        _ tensorName: String,
        metadata: [String: String]
    ) -> String? {
        MLXSafetensorsTensorMetadata.dtype(for: tensorName, in: metadata)?.uppercased()
    }
}
