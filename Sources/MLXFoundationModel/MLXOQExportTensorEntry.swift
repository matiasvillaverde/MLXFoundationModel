import Foundation

/// One source tensor and its expected output tensors for an oQ export.
public struct MLXOQExportTensorEntry: Codable, Equatable, Hashable, Sendable {
    public let disposition: MLXOQExportTensorDisposition
    public let dtype: String?
    public let outputNames: [String]
    public let quantizationSpec: MLXOQQuantizationSpec?
    public let shape: [Int]
    public let sourceFilename: String
    public let sourceName: String

    public init(
        sourceName: String,
        sourceFilename: String,
        dtype: String?,
        shape: [Int],
        disposition: MLXOQExportTensorDisposition,
        quantizationSpec: MLXOQQuantizationSpec?,
        outputNames: [String]
    ) {
        self.disposition = disposition
        self.dtype = dtype
        self.outputNames = outputNames
        self.quantizationSpec = quantizationSpec
        self.shape = shape
        self.sourceFilename = sourceFilename
        self.sourceName = sourceName
    }

    public var isQuantized: Bool {
        disposition == .quantize
    }
}
