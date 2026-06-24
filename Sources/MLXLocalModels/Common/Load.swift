import Foundation
import Hub
import MLX
import MLXNN
import OSLog

internal enum ModelLoadError: LocalizedError, Equatable {
    case cannotEnumerateWeights(URL)

    internal var errorDescription: String? {
        switch self {
        case .cannotEnumerateWeights(let directory):
            "Could not enumerate model weights in '\(directory.path())'."
        }
    }
}

internal enum ModelArtifactMatcher {
    internal static let snapshotPatterns = [
        "*.safetensors",
        "*.json",
        "*.jinja",
        "merges.txt",
        "tokenizer.model",
        "vocab.*"
    ]
}

internal enum SafetensorFileDiscovery {
    internal static func safetensorURLs(
        in directory: URL,
        fileManager: FileManager = .default
    ) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path(), isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ModelLoadError.cannotEnumerateWeights(directory)
        }

        guard let enumerator = fileManager.enumerator(
            at: directory,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            throw ModelLoadError.cannotEnumerateWeights(directory)
        }

        var urls: [URL] = []
        for case let url as URL in enumerator
        where url.pathExtension.lowercased() == "safetensors" {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey])
            if values.isRegularFile == true {
                urls.append(url)
            }
        }

        return urls.sorted { lhs, rhs in
            lhs.path < rhs.path
        }
    }
}

private struct LoadedWeights {
    var arrays: [String: MLXArray] = [:]
    var metadata: [String: String] = [:]
    var fileCount = 0

    mutating func mergeFile(
        arrays newArrays: [String: MLXArray],
        dtypeMetadata: [String: String],
        fileMetadata: [String: String]
    ) {
        for (key, value) in newArrays {
            arrays[key] = value
        }
        metadata.merge(dtypeMetadata) { current, _ in current }
        metadata.merge(fileMetadata) { current, _ in current }
        fileCount += 1
    }
}

internal func downloadModel(
    hub: HubApi,
    configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    let logger = MLXObservability.logger(for: .modelLoad)

    do {
        switch configuration.id {
        case .id(let id, let revision):
            logger.info("Downloading model: \(id) (revision: \(revision))")
            let url = try await hub.snapshot(
                from: Hub.Repo(id: id),
                revision: revision,
                matching: ModelArtifactMatcher.snapshotPatterns,
                progressHandler: progressHandler
            )
            logger.info("Model downloaded successfully to: \(url.path)")
            return url

        case .directory(let directory):
            logger.debug("Using local model directory: \(directory.path)")
            return directory
        }
    } catch Hub.HubClientError.authorizationRequired {
        logger.warning("Authorization required or model not found, falling back to local directory")
        return configuration.modelDirectory(hub: hub)
    } catch {
        guard ModelDownloadFallback.shouldUseLocalDirectory(for: error) else {
            logger.error("Failed to download model: \(error.localizedDescription)")
            throw error
        }

        logger.warning("No internet connection, using local model directory")
        return configuration.modelDirectory(hub: hub)
    }
}

private enum ModelDownloadFallback {
    static func shouldUseLocalDirectory(for error: Error) -> Bool {
        let nsError = error as NSError
        return nsError.domain == NSURLErrorDomain
            && nsError.code == NSURLErrorNotConnectedToInternet
    }
}

internal func loadWeights(
    modelDirectory: URL,
    model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    let logger = MLXObservability.logger(for: .modelLoad)
    logger.info("Loading model weights from: \(modelDirectory.path)")

    let loadedWeights = try loadSafetensorWeights(from: modelDirectory, logger: logger)
    logger.debug(
        "Loaded \(loadedWeights.fileCount) safetensor files with \(loadedWeights.arrays.count) total weights"
    )

    let packedScaleResult = MLXQuantizedWeightSanitizer.sanitizePackedScalePairs(
        loadedWeights.arrays,
        metadata: loadedWeights.metadata
    )
    if packedScaleResult.report.packedScaleCount > 0 {
        logger.info(
            "Dequantized \(packedScaleResult.report.packedScaleCount) FP8 packed scale pairs"
        )
    }

    let weights = model.sanitize(weights: packedScaleResult.weights, metadata: loadedWeights.metadata)
    if quantization != nil || perLayerQuantization != nil {
        logger.debug("Applying quantization to model")
        quantize(model: model) { path, _ in
            guard weights["\(path).scales"] != nil else {
                return nil
            }
            return perLayerQuantization?.quantization(layer: path)?.asTuple ?? quantization?.asTuple
        }
    }

    try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
    eval(model)
    logger.info("Model weights loaded successfully")
}

private func loadSafetensorWeights(from directory: URL, logger: Logger) throws -> LoadedWeights {
    var loadedWeights = LoadedWeights()
    for url in try SafetensorFileDiscovery.safetensorURLs(in: directory) {
        logger.debug("Loading weights from: \(url.lastPathComponent)")
        let (arrays, fileMetadata) = try loadArraysAndMetadata(url: url)
        loadedWeights.mergeFile(
            arrays: arrays,
            dtypeMetadata: MLXSafetensorsTensorMetadata.encodedDTypeMetadata(at: url),
            fileMetadata: fileMetadata
        )
    }
    return loadedWeights
}
