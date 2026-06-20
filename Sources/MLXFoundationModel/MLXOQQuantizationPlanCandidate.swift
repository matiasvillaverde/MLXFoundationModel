import Foundation

struct MLXOQQuantizationPlanCandidate {
    let score: Double
    let spec: MLXOQQuantizationSpec
    let tensor: MLXOQTensorDescriptor
}
