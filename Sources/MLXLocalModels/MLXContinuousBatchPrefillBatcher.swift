internal enum MLXContinuousBatchPrefillBatcher {
    internal static func groups(
        for requests: [MLXContinuousBatchPrefillRequest]
    ) throws -> [[MLXContinuousBatchPrefillRequest]] {
        var groups: [Group] = []
        groups.reserveCapacity(requests.count)

        for index in requests.indices {
            let request = requests[index]
            guard !request.promptTokenIDs.isEmpty else {
                throw MLXContinuousBatchPrefillError.emptyPrompt(rowIndex: index)
            }

            let key = GroupKey(request: request)
            if let groupIndex = groups.firstIndex(where: { $0.key == key }) {
                groups[groupIndex].requests.append(request)
            } else {
                groups.append(Group(key: key, requests: [request]))
            }
        }

        return groups.map(\.requests)
    }

    internal static func groups(
        forQueued queuedRequests: [MLXContinuousBatchQueuedPrefillRequest]
    ) throws -> [[MLXContinuousBatchQueuedPrefillRequest]] {
        var groups: [QueuedGroup] = []
        groups.reserveCapacity(queuedRequests.count)

        for index in queuedRequests.indices {
            let queuedRequest = queuedRequests[index]
            guard !queuedRequest.request.promptTokenIDs.isEmpty else {
                throw MLXContinuousBatchPrefillError.emptyPrompt(rowIndex: index)
            }

            let key = GroupKey(request: queuedRequest.request)
            if let groupIndex = groups.firstIndex(where: { $0.key == key }) {
                groups[groupIndex].requests.append(queuedRequest)
            } else {
                groups.append(QueuedGroup(key: key, requests: [queuedRequest]))
            }
        }

        return groups.map(\.requests)
    }
}
