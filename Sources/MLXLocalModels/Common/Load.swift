// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXNN
import OSLog
import Tokenizers

/// Download the model using the `HubApi`.
///
/// This will download `*.safetensors` and `*.json` if the ``ModelConfiguration``
/// represents a Hub id, e.g. `mlx-community/gemma-2-2b-it-4bit`.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``
///
/// - Parameters:
///   - hub: HubApi instance
///   - configuration: the model identifier
///   - progressHandler: callback for progress
/// - Returns: URL for the directory containing downloaded files
internal func downloadModel(
    hub: HubApi, configuration: ModelConfiguration,
    progressHandler: @Sendable @escaping (Progress) -> Void
) async throws -> URL {
    let logger = MLXObservability.logger(for: .modelLoad)

    do {
        switch configuration.id {
        case .id(let id, let revision):
            logger.info("Downloading model: \(id) (revision: \(revision))")
            // download the model weights
            let repo = Hub.Repo(id: id)
            let modelFiles = [
                "*.safetensors",
                "*.json",
                "*.jinja",
                "merges.txt",
                "tokenizer.model",
                "vocab.*"
            ]
            let url = try await hub.snapshot(
                from: repo,
                revision: revision,
                matching: modelFiles,
                progressHandler: progressHandler
            )
            logger.info("Model downloaded successfully to: \(url.path)")
            return url
        case .directory(let directory):
            logger.debug("Using local model directory: \(directory.path)")
            return directory
        }

    } catch Hub.HubClientError.authorizationRequired {
        // an authorizationRequired means (typically) that the named repo doesn't exist on
        // on the server so retry with local only configuration
        logger.warning("Authorization required or model not found, falling back to local directory")
        return configuration.modelDirectory(hub: hub)

    } catch {
        let nserror = error as NSError
        if nserror.domain == NSURLErrorDomain && nserror.code == NSURLErrorNotConnectedToInternet {
            // Error Domain=NSURLErrorDomain Code=-1009 "The Internet connection appears to be offline."
            // fall back to the local directory
            logger.warning("No internet connection, using local model directory")
            return configuration.modelDirectory(hub: hub)
        } else {
            logger.error("Failed to download model: \(error.localizedDescription)")
            throw error
        }
    }
}

/// Load model weights.
///
/// This is typically called via ``ModelFactory/load(hub:configuration:progressHandler:)``.
/// This function loads all `safetensor` files in the given `modelDirectory`,
/// calls ``LanguageModel/sanitize(weights:)``, applies optional quantization, and
/// updates the model with the weights.
internal func loadWeights(
    modelDirectory: URL, model: LanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    let logger = MLXObservability.logger(for: .modelLoad)
    logger.info("Loading model weights from: \(modelDirectory.path)")

    // load the weights
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    var fileCount = 0
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    for case let url as URL in enumerator {
        if url.pathExtension == "safetensors" {
            logger.debug("Loading weights from: \(url.lastPathComponent)")
            let (w, fileMetadata) = try loadArraysAndMetadata(url: url)
            for (key, value) in w {
                weights[key] = value
            }
            metadata.merge(
                MLXSafetensorsTensorMetadata.encodedDTypeMetadata(at: url)
            ) { current, _ in current }
            metadata.merge(fileMetadata) { current, _ in current }
            fileCount += 1
        }
    }
    logger.debug("Loaded \(fileCount) safetensor files with \(weights.count) total weights")

    let packedScaleResult = MLXQuantizedWeightSanitizer.sanitizePackedScalePairs(
        weights,
        metadata: metadata
    )
    if packedScaleResult.report.packedScaleCount > 0 {
        logger.info(
            "Dequantized \(packedScaleResult.report.packedScaleCount) FP8 packed scale pairs"
        )
    }
    weights = packedScaleResult.weights

    // per-model cleanup
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        logger.debug("Applying quantization to model")
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
    logger.info("Model weights loaded successfully")
}
