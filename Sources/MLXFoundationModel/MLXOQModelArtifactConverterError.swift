import Foundation

/// Errors raised while writing an oQ artifact directory.
public enum MLXOQModelArtifactConverterError: Error, Equatable, Sendable {
    case inputAndOutputDirectoryMatch(URL)
    case invalidAuxiliaryFile(URL)
    case invalidManifestEntry(String)
    case invalidQuantizationMode(String)
    case missingQuantizationSpec(String)
    case missingQuantizedBiases(String)
    case missingSourceTensor(String, String)
    case outputDirectoryExists(URL)
    case outputDirectoryInsideModelDirectory(URL)
}
