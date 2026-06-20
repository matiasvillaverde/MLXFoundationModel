import Foundation

struct MLXOQQuantizationRuleContext {
    let tensor: MLXOQTensorDescriptor
    let level: MLXOQLevel
    let traits: MLXOQModelQuantizationTraits
    let defaultGroupSize: Int

    var name: String {
        normalizedName(tensor.name)
    }

    var lowercasedName: String {
        name.lowercased()
    }

    var layerIndex: Int? {
        MLXOQLayerIndexParser.layerIndex(in: name)
    }

    var isMTPProtected: Bool {
        guard name.hasPrefix("mtp.") || name.contains(".mtp.") else {
            return false
        }
        return name.hasSuffix("mtp.fc.weight") ||
            name.contains(".mtp.fc.weight") ||
            name.hasSuffix(".e_proj.weight") ||
            name.hasSuffix(".h_proj.weight") ||
            name.contains(".hc_head.") ||
            name.hasSuffix(".hc_head_fn") ||
            name.hasSuffix(".hc_head_base") ||
            name.hasSuffix(".hc_head_scale")
    }

    var isSensitiveLayer: Bool {
        guard let layerIndex else {
            return false
        }
        return layerIndex < traits.numLayers / 8 ||
            layerIndex >= 7 * traits.numLayers / 8
    }

    var isRoutedExpert: Bool {
        lowercasedName.contains("switch_mlp") ||
            lowercasedName.contains("block_sparse_moe") ||
            (lowercasedName.contains("experts") && !lowercasedName.contains("shared_expert"))
    }

    func spec(bits requestedBits: Int) -> MLXOQQuantizationSpec {
        MLXOQQuantizationSpec(
            bits: max(requestedBits, level.baseBits),
            groupSize: defaultGroupSize
        )
    }

    private func normalizedName(_ value: String) -> String {
        if value.hasSuffix(".scales") || value.hasSuffix(".biases") {
            return String(value.dropLast(7))
        }
        return value
    }
}
