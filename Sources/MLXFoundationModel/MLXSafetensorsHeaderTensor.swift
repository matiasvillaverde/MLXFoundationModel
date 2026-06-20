import Foundation

struct MLXSafetensorsHeaderTensor: Equatable, Sendable {
    let dtype: String?
    let name: String
    let shape: [Int]
    let sourceFilename: String

    var oQDescriptor: MLXOQTensorDescriptor? {
        guard name.hasSuffix(".weight"),
            shape.count >= 2 else {
            return nil
        }
        return MLXOQTensorDescriptor(name: name, shape: shape)
    }
}
