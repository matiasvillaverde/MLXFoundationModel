internal struct QueuedGroup {
    let key: GroupKey
    var requests: [MLXContinuousBatchQueuedPrefillRequest]
}
