internal enum MLXGenerationExecutionPlanReason: Sendable, Equatable, Hashable {
    case continuousBatchingSelected
    case continuousBatchingUnsupported
    case nativeMTPRequiresScalar
    case scalarRequested
    case sharedKVMTPRequiresScalar
    case specPrefillRequiresScalar
    case speculativeDecodingRequiresScalar
}
