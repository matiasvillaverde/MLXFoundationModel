import Foundation

/// Factory for creating MLX-backed generation sessions.
public enum MLXSessionFactory {
    /// Create a new local MLX generation session.
    /// - Returns: A generation session backed by MLX.
    public static func create() -> any MLXGeneratingSession {
        MLXSession()
    }
}
