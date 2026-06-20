import Foundation

extension MLXOQModelArtifactConverter {
    static func writeConvertedConfig(
        sourceDirectory: URL,
        outputDirectory: URL,
        manifest: MLXOQExportManifest
    ) throws {
        var config = try readConfig(sourceDirectory.appendingPathComponent("config.json"))
        let quantization = quantizationConfig(for: manifest)
        if !quantization.isEmpty {
            config["quantization"] = quantization
        }
        config["mlx_oq"] = oQMetadata(for: manifest)
        let url = outputDirectory.appendingPathComponent("config.json")
        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url)
    }

    private static func readConfig(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXOQModelArtifactPlannerError.missingConfig(url)
        }
        return config
    }

    private static func quantizationConfig(for manifest: MLXOQExportManifest) -> [String: Any] {
        let entries = manifest.entries.filter(\.isQuantized)
        guard let defaultSpec = entries.compactMap(\.quantizationSpec).first else {
            return [:]
        }
        var config = quantizationObject(for: defaultSpec)
        for entry in entries {
            guard let spec = entry.quantizationSpec else {
                continue
            }
            config[moduleName(for: entry.sourceName)] = quantizationObject(for: spec)
        }
        return config
    }

    private static func quantizationObject(for spec: MLXOQQuantizationSpec) -> [String: Any] {
        [
            "bits": spec.bits,
            "group_size": spec.groupSize,
            "quantization_mode": spec.mode
        ]
    }

    private static func oQMetadata(for manifest: MLXOQExportManifest) -> [String: Any] {
        [
            "baseline_bits_per_weight": manifest.baselineBitsPerWeight,
            "effective_bits_per_weight": manifest.effectiveBitsPerWeight,
            "estimated_serialized_bytes": manifest.estimatedSerializedBytes,
            "level": manifest.level
        ]
    }
}
