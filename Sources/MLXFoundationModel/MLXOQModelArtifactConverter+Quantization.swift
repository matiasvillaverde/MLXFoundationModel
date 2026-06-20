import Foundation
import MLX

extension MLXOQModelArtifactConverter {
    static func convertedArrays(
        for entry: MLXOQExportTensorEntry,
        sourceArray: MLXArray
    ) throws -> [String: MLXArray] {
        switch entry.disposition {
        case .copyFullPrecision:
            return [entry.sourceName: sourceArray]

        case .quantize:
            return try quantizedArrays(for: entry, sourceArray: sourceArray)
        }
    }

    static func moduleName(for tensorName: String) -> String {
        if tensorName.hasSuffix(".weight") {
            return String(tensorName.dropLast(".weight".count))
        }
        return tensorName
    }

    private static func quantizedArrays(
        for entry: MLXOQExportTensorEntry,
        sourceArray: MLXArray
    ) throws -> [String: MLXArray] {
        guard let spec = entry.quantizationSpec else {
            throw MLXOQModelArtifactConverterError.missingQuantizationSpec(entry.sourceName)
        }
        guard let mode = QuantizationMode(rawValue: spec.mode) else {
            throw MLXOQModelArtifactConverterError.invalidQuantizationMode(spec.mode)
        }
        let quantized = MLX.quantized(
            sourceArray,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: mode
        )
        return try outputArrays(
            for: entry,
            quantized: MLXOQQuantizedTensorArrays(
                weight: quantized.wq,
                scales: quantized.scales,
                biases: quantized.biases
            )
        )
    }

    private static func outputArrays(
        for entry: MLXOQExportTensorEntry,
        quantized: MLXOQQuantizedTensorArrays
    ) throws -> [String: MLXArray] {
        guard entry.outputNames.count >= 2 else {
            throw MLXOQModelArtifactConverterError.invalidManifestEntry(entry.sourceName)
        }
        var arrays = [
            entry.outputNames[0]: quantized.weight,
            entry.outputNames[1]: quantized.scales
        ]
        if entry.outputNames.count > 2 {
            arrays[entry.outputNames[2]] = try requiredBiases(quantized.biases, entry: entry)
        }
        return arrays
    }

    private static func requiredBiases(
        _ biases: MLXArray?,
        entry: MLXOQExportTensorEntry
    ) throws -> MLXArray {
        guard let biases else {
            throw MLXOQModelArtifactConverterError.missingQuantizedBiases(entry.sourceName)
        }
        return biases
    }
}
