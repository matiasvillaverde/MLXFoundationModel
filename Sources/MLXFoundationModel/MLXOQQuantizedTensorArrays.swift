import Foundation
import MLX

struct MLXOQQuantizedTensorArrays {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
}
