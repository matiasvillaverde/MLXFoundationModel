internal struct MLXContinuousBatchRowState<
    Processor: Sendable,
    Sampler: Sendable,
    Cache: Sendable
>: Sendable {
    internal var processor: Processor
    internal var sampler: Sampler
    internal var cache: Cache
    internal var generatedTokenCount: Int

    internal init(
        processor: Processor,
        sampler: Sampler,
        cache: Cache,
        generatedTokenCount: Int = 0
    ) {
        self.processor = processor
        self.sampler = sampler
        self.cache = cache
        self.generatedTokenCount = max(0, generatedTokenCount)
    }
}
