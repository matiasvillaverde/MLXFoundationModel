import Foundation

extension MLXOQModelArtifactConverter {
    static func prepareOutputDirectory(
        sourceDirectory: URL,
        outputDirectory: URL,
        options: MLXOQModelArtifactConverterOptions
    ) throws {
        try validateOutputDirectory(sourceDirectory: sourceDirectory, outputDirectory: outputDirectory)
        if FileManager.default.fileExists(atPath: outputDirectory.path) {
            guard options.overwriteOutputDirectory else {
                throw MLXOQModelArtifactConverterError.outputDirectoryExists(outputDirectory)
            }
            try FileManager.default.removeItem(at: outputDirectory)
        }
        try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
    }

    static func copyAuxiliaryFiles(from sourceDirectory: URL, to outputDirectory: URL) throws {
        guard let enumerator = FileManager.default.enumerator(atPath: sourceDirectory.path) else {
            return
        }
        for case let relativePath as String in enumerator {
            let url = sourceDirectory.appendingPathComponent(relativePath)
            guard try shouldCopyAuxiliaryFile(url) else {
                continue
            }
            try copyAuxiliaryFile(
                relativePath,
                sourceDirectory: sourceDirectory,
                outputDirectory: outputDirectory
            )
        }
    }

    private static func validateOutputDirectory(
        sourceDirectory: URL,
        outputDirectory: URL
    ) throws {
        let sourcePath = normalizedDirectoryPath(sourceDirectory)
        let outputPath = normalizedDirectoryPath(outputDirectory)
        if sourcePath == outputPath {
            throw MLXOQModelArtifactConverterError.inputAndOutputDirectoryMatch(outputDirectory)
        }
        if outputPath.hasPrefix(sourcePath + "/") {
            throw MLXOQModelArtifactConverterError.outputDirectoryInsideModelDirectory(outputDirectory)
        }
    }

    private static func shouldCopyAuxiliaryFile(_ url: URL) throws -> Bool {
        let values = try url.resourceValues(forKeys: [.isRegularFileKey])
        guard values.isRegularFile == true else {
            return false
        }
        return url.lastPathComponent != "config.json" && url.pathExtension != "safetensors"
    }

    private static func copyAuxiliaryFile(
        _ relativePath: String,
        sourceDirectory: URL,
        outputDirectory: URL
    ) throws {
        let url = sourceDirectory.appendingPathComponent(relativePath)
        let destination = outputDirectory.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try FileManager.default.copyItem(at: url, to: destination)
    }

    private static func normalizedDirectoryPath(_ url: URL) -> String {
        canonicalPath(url.standardizedFileURL.resolvingSymlinksInPath().path(percentEncoded: false))
    }

    private static func canonicalPath(_ path: String) -> String {
        if path.hasPrefix("/private/var/") {
            return String(path.dropFirst("/private".count))
        }
        return path
    }
}
