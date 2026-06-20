import Foundation

/// Builds oQ quantization plans from a downloaded MLX model directory.
public enum MLXOQModelArtifactPlanner {
    /// Builds a budgeted oQ plan from `config.json` and safetensors headers.
    public static func plan(
        modelDirectory: URL,
        level: MLXOQLevel,
        options: MLXOQQuantizationPlanOptions = .init()
    ) throws -> MLXOQQuantizationPlan {
        let config = try config(in: modelDirectory)
        return try plan(
            config: config,
            tensors: tensors(in: modelDirectory),
            modelDirectory: modelDirectory,
            level: level,
            options: options
        )
    }

    /// Builds a budgeted oQ plan after parsing an oQ level string such as `oQ4`.
    public static func plan(
        modelDirectory: URL,
        level: String,
        options: MLXOQQuantizationPlanOptions = .init()
    ) throws -> MLXOQQuantizationPlan {
        guard let parsedLevel = MLXOQLevel(level) else {
            throw MLXOQModelArtifactPlannerError.invalidOQLevel(level)
        }
        return try plan(modelDirectory: modelDirectory, level: parsedLevel, options: options)
    }

    /// Builds a header-only export manifest for a future streaming oQ conversion.
    public static func exportManifest(
        modelDirectory: URL,
        level: MLXOQLevel,
        options: MLXOQQuantizationPlanOptions = .init()
    ) throws -> MLXOQExportManifest {
        let config = try config(in: modelDirectory)
        let tensors = try tensors(in: modelDirectory)
        let plan = try plan(
            config: config,
            tensors: tensors,
            modelDirectory: modelDirectory,
            level: level,
            options: options
        )
        return MLXOQExportManifestBuilder.manifest(tensors: tensors, plan: plan)
    }

    /// Builds a header-only export manifest after parsing an oQ level string.
    public static func exportManifest(
        modelDirectory: URL,
        level: String,
        options: MLXOQQuantizationPlanOptions = .init()
    ) throws -> MLXOQExportManifest {
        guard let parsedLevel = MLXOQLevel(level) else {
            throw MLXOQModelArtifactPlannerError.invalidOQLevel(level)
        }
        return try exportManifest(modelDirectory: modelDirectory, level: parsedLevel, options: options)
    }

    private static func config(in modelDirectory: URL) throws -> [String: Any] {
        let url = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MLXOQModelArtifactPlannerError.missingConfig(url)
        }
        let data = try Data(contentsOf: url)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw MLXOQModelArtifactPlannerError.missingConfig(url)
        }
        return config
    }

    private static func plan(
        config: [String: Any],
        tensors: [MLXSafetensorsHeaderTensor],
        modelDirectory: URL,
        level: MLXOQLevel,
        options: MLXOQQuantizationPlanOptions
    ) throws -> MLXOQQuantizationPlan {
        let descriptors = try tensorDescriptors(tensors, modelDirectory: modelDirectory)
        let planner = MLXOQQuantizationPlanner(
            level: level,
            traits: MLXOQModelQuantizationTraits.make(config: config)
        )
        return planner.plan(for: descriptors, options: options)
    }

    private static func tensors(
        in modelDirectory: URL
    ) throws -> [MLXSafetensorsHeaderTensor] {
        try MLXSafetensorsHeaderScanner.tensors(in: modelDirectory)
    }

    private static func tensorDescriptors(
        _ tensors: [MLXSafetensorsHeaderTensor],
        modelDirectory: URL
    ) throws -> [MLXOQTensorDescriptor] {
        let descriptors = tensors.compactMap(\.oQDescriptor)
        guard !descriptors.isEmpty else {
            throw MLXOQModelArtifactPlannerError.noSafetensorsTensors(modelDirectory)
        }
        return descriptors
    }
}
