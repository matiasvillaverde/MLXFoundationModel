import MLXLocalModels

/// Factory closure used by ``MLXModelPool`` to create local generation sessions.
public typealias MLXModelPoolSessionFactory = @Sendable () -> any MLXGeneratingSession
