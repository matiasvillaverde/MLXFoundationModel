import Foundation
import MLX

/// Converts a model directory into an MLX-compatible oQ artifact directory.
public enum MLXOQModelArtifactConverter {
    /// Plans and writes an oQ artifact directory for a typed oQ level.
    @discardableResult
    public static func convert(
        modelDirectory: URL,
        outputDirectory: URL,
        level: MLXOQLevel,
        planOptions: MLXOQQuantizationPlanOptions = .init(),
        converterOptions: MLXOQModelArtifactConverterOptions = .init()
    ) throws -> MLXOQExportManifest {
        let manifest = try MLXOQModelArtifactPlanner.exportManifest(
            modelDirectory: modelDirectory,
            level: level,
            options: planOptions
        )
        try convert(
            modelDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            manifest: manifest,
            options: converterOptions
        )
        return manifest
    }

    /// Plans and writes an oQ artifact directory for a level string such as `oQ4`.
    @discardableResult
    public static func convert(
        modelDirectory: URL,
        outputDirectory: URL,
        level: String,
        planOptions: MLXOQQuantizationPlanOptions = .init(),
        converterOptions: MLXOQModelArtifactConverterOptions = .init()
    ) throws -> MLXOQExportManifest {
        guard let parsedLevel = MLXOQLevel(level) else {
            throw MLXOQModelArtifactPlannerError.invalidOQLevel(level)
        }
        return try convert(
            modelDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            level: parsedLevel,
            planOptions: planOptions,
            converterOptions: converterOptions
        )
    }

    /// Writes an oQ artifact directory from a previously built manifest.
    public static func convert(
        modelDirectory: URL,
        outputDirectory: URL,
        manifest: MLXOQExportManifest,
        options: MLXOQModelArtifactConverterOptions = .init()
    ) throws {
        try prepareOutputDirectory(
            sourceDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            options: options
        )
        if options.copyAuxiliaryFiles {
            try copyAuxiliaryFiles(from: modelDirectory, to: outputDirectory)
        }
        try writeConvertedConfig(
            sourceDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            manifest: manifest
        )
        try writeSafetensorShards(
            sourceDirectory: modelDirectory,
            outputDirectory: outputDirectory,
            manifest: manifest
        )
    }
}
