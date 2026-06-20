import Foundation

/// Filesystem options for oQ artifact conversion.
public struct MLXOQModelArtifactConverterOptions: Equatable, Sendable {
    public let copyAuxiliaryFiles: Bool
    public let overwriteOutputDirectory: Bool

    public init(
        overwriteOutputDirectory: Bool = false,
        copyAuxiliaryFiles: Bool = true
    ) {
        self.copyAuxiliaryFiles = copyAuxiliaryFiles
        self.overwriteOutputDirectory = overwriteOutputDirectory
    }
}
