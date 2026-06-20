import Foundation

/// Conversion action for one source tensor in an oQ export manifest.
public enum MLXOQExportTensorDisposition: String, Codable, Equatable, Hashable, Sendable {
    case copyFullPrecision = "copy_full_precision"
    case quantize = "quantize"
}
