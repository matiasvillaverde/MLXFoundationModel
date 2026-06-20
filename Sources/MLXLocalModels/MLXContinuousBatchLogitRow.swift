import MLX

internal struct MLXContinuousBatchLogitRow: Sendable {
    internal var processor: LogitProcessor?
    internal var sampler: LogitSampler
    internal var generatedTokenCount: Int

    internal init(
        processor: LogitProcessor?,
        sampler: LogitSampler,
        generatedTokenCount: Int = 0
    ) {
        self.processor = processor
        self.sampler = sampler
        self.generatedTokenCount = max(0, generatedTokenCount)
    }

    internal mutating func sample(logits: MLXArray) -> MLXArray {
        var logits = logits
        logits = processor?.process(logits: logits) ?? logits
        let token = sampler.sample(logits: logits)
        processor?.didSample(token: token)
        generatedTokenCount += 1
        return token
    }
}
