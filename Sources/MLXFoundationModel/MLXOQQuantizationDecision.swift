import Foundation

/// oQ decision for a single tensor.
public enum MLXOQQuantizationDecision: Equatable, Hashable, Sendable {
    case keepFullPrecision
    case quantize(MLXOQQuantizationSpec)

    public var quantizationSpec: MLXOQQuantizationSpec? {
        guard case .quantize(let spec) = self else {
            return nil
        }
        return spec
    }
}
