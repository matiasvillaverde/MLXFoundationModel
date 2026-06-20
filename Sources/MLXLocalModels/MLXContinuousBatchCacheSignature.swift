internal struct MLXContinuousBatchCacheSignature: Equatable, Sendable {
    internal let maxKVSize: Int?
    internal let kvBits: Int?
    internal let kvGroupSize: Int
    internal let quantizedKVStart: Int
    internal let prefillStepSize: Int

    internal init(parameters: GenerateParameters) {
        self.maxKVSize = parameters.maxKVSize
        self.kvBits = parameters.kvBits
        self.kvGroupSize = parameters.kvGroupSize
        self.quantizedKVStart = parameters.quantizedKVStart
        self.prefillStepSize = parameters.prefillStepSize
    }
}
