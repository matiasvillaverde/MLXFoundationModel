import Foundation
import MLX

extension MLXOQModelArtifactConverter {
    static func writeSafetensorShards(
        sourceDirectory: URL,
        outputDirectory: URL,
        manifest: MLXOQExportManifest
    ) throws {
        let entriesByShard = Dictionary(grouping: manifest.entries, by: \.sourceFilename)
        for filename in entriesByShard.keys.sorted() {
            guard let entries = entriesByShard[filename] else {
                continue
            }
            try writeSafetensorShard(
                filename: filename,
                entries: entries,
                sourceDirectory: sourceDirectory,
                outputDirectory: outputDirectory
            )
        }
    }

    private static func writeSafetensorShard(
        filename: String,
        entries: [MLXOQExportTensorEntry],
        sourceDirectory: URL,
        outputDirectory: URL
    ) throws {
        let sourceURL = sourceDirectory.appendingPathComponent(filename)
        let outputURL = outputDirectory.appendingPathComponent(filename)
        let loaded = try MLX.loadArraysAndMetadata(url: sourceURL)
        let arrays = try outputArrays(from: loaded.0, entries: entries)
        MLX.eval(arrays.values)
        try MLX.save(arrays: arrays, metadata: loaded.1, url: outputURL)
    }

    private static func outputArrays(
        from sourceArrays: [String: MLXArray],
        entries: [MLXOQExportTensorEntry]
    ) throws -> [String: MLXArray] {
        var outputArrays = sourceArrays
        for entry in entries {
            let sourceArray = try sourceArray(for: entry, in: sourceArrays)
            removeGeneratedOutputs(for: entry, from: &outputArrays)
            outputArrays.merge(try convertedArrays(for: entry, sourceArray: sourceArray)) { _, new in new }
        }
        return outputArrays
    }

    private static func sourceArray(
        for entry: MLXOQExportTensorEntry,
        in sourceArrays: [String: MLXArray]
    ) throws -> MLXArray {
        guard let sourceArray = sourceArrays[entry.sourceName] else {
            throw MLXOQModelArtifactConverterError.missingSourceTensor(
                entry.sourceName,
                entry.sourceFilename
            )
        }
        return sourceArray
    }

    private static func removeGeneratedOutputs(
        for entry: MLXOQExportTensorEntry,
        from outputArrays: inout [String: MLXArray]
    ) {
        outputArrays[entry.sourceName] = nil
        for outputName in entry.outputNames {
            outputArrays[outputName] = nil
        }
    }
}
