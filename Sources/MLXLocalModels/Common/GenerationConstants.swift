/// Shared defaults for generation, cache rotation, and sampling processors.
internal enum GenerationConstants {
    /// Maximum prompt tokens evaluated per default prefill chunk.
    static let defaultPrefillStepSize = 512

    /// Recent-token window used by repetition, presence, and frequency penalties.
    static let defaultRepetitionContextSize = 20

    /// Default lookback used by legacy repetition-penalty range settings.
    static let defaultRepetitionPenaltyRange = 64

    /// Prompt-prefix tokens retained when rotating a bounded KV cache.
    static let rotatingCacheKeepTokens = 4

    /// Default token grouping for runtime KV-cache quantization.
    static let defaultKVCacheGroupSize = 64

    /// Number of recent decoded text segments scanned for stop strings.
    static let stopSequenceCheckWindowSize = 10
}
