import Foundation

enum MLXOQExportManifestBuilder {
    static func manifest(
        tensors: [MLXSafetensorsHeaderTensor],
        plan: MLXOQQuantizationPlan
    ) -> MLXOQExportManifest {
        MLXOQExportManifest(
            level: plan.level.label,
            entries: entries(tensors: tensors, decisions: plan.decisions),
            baselineBitsPerWeight: plan.baselineBitsPerWeight,
            effectiveBitsPerWeight: plan.effectiveBitsPerWeight,
            estimatedSerializedBytes: plan.estimatedSerializedBytes
        )
    }

    private static func entries(
        tensors: [MLXSafetensorsHeaderTensor],
        decisions: [String: MLXOQQuantizationDecision]
    ) -> [MLXOQExportTensorEntry] {
        tensors
            .sorted { left, right in
                if left.sourceFilename == right.sourceFilename {
                    return left.name < right.name
                }
                return left.sourceFilename < right.sourceFilename
            }
            .map { tensor in
                entry(tensor: tensor, decision: decisions[tensor.name])
            }
    }

    private static func entry(
        tensor: MLXSafetensorsHeaderTensor,
        decision: MLXOQQuantizationDecision?
    ) -> MLXOQExportTensorEntry {
        guard let spec = decision?.quantizationSpec else {
            return MLXOQExportTensorEntry(
                sourceName: tensor.name,
                sourceFilename: tensor.sourceFilename,
                dtype: tensor.dtype,
                shape: tensor.shape,
                disposition: .copyFullPrecision,
                quantizationSpec: nil,
                outputNames: [tensor.name]
            )
        }
        return MLXOQExportTensorEntry(
            sourceName: tensor.name,
            sourceFilename: tensor.sourceFilename,
            dtype: tensor.dtype,
            shape: tensor.shape,
            disposition: .quantize,
            quantizationSpec: spec,
            outputNames: outputNames(for: tensor.name, spec: spec)
        )
    }

    private static func outputNames(
        for tensorName: String,
        spec: MLXOQQuantizationSpec
    ) -> [String] {
        let prefix = tensorName.hasSuffix(".weight")
            ? String(tensorName.dropLast(".weight".count))
            : tensorName
        var names = [
            "\(prefix).weight",
            "\(prefix).scales"
        ]
        if spec.mode == "affine" {
            names.append("\(prefix).biases")
        }
        return names
    }
}
