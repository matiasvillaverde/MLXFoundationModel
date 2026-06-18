// Copyright © 2024 Apple Inc.

import Foundation

/// Constants used throughout the MLX generation pipeline
///
/// These constants define default values for prompt processing, token generation,
/// cache management, and other core generation parameters.
internal enum GenerationConstants {
    // MARK: - Prompt Processing

    /// Default step size for prompt prefill operations
    ///
    /// This value determines how many tokens are processed at once during the initial
    /// prompt evaluation phase. Larger values can improve throughput but may increase
    /// memory usage. A value of 512 provides a good balance between performance and
    /// memory efficiency for most models.
    static let defaultPrefillStepSize = 512

    // MARK: - Repetition Penalty

    /// Default context size for repetition penalty calculation
    ///
    /// Specifies how many recent tokens to consider when applying repetition penalties.
    /// A value of 20 tokens provides sufficient context to prevent immediate repetition
    /// while keeping the computational cost reasonable.
    static let defaultRepetitionContextSize = 20

    /// Default range for repetition penalty application
    ///
    /// When a repetition penalty range is not explicitly specified, this value is used
    /// to determine how far back in the token sequence to look for repeated tokens.
    /// A value of 64 tokens balances repetition detection with performance.
    static let defaultRepetitionPenaltyRange = 64

    // MARK: - KV Cache Management

    /// Number of tokens to preserve when rotating the KV cache
    ///
    /// When the rotating cache reaches its maximum size, it keeps the first N tokens
    /// (typically important prompt tokens) and overwrites older generated tokens.
    /// A value of 4 preserves critical context while allowing efficient cache rotation.
    static let rotatingCacheKeepTokens = 4

    /// Default group size for KV cache quantization
    ///
    /// Used when quantizing key-value cache entries to reduce memory usage.
    /// Tokens are grouped together for quantization to maintain quality while
    /// achieving compression. A value of 64 provides good compression with minimal
    /// quality loss for most models.
    static let defaultKVCacheGroupSize = 64

    // MARK: - Stop Sequence Detection

    /// Number of recent text segments to check for stop sequences
    ///
    /// To optimize performance, stop sequence detection only examines the most recent
    /// text segments rather than re-scanning the entire generated text. This value
    /// determines how many recent segments are checked. A value of 10 is sufficient
    /// because stop sequences are typically short (< 20 characters) and will appear
    /// within the most recent segments.
    static let stopSequenceCheckWindowSize = 10
}
