internal struct MLXContinuousBatchGenerationBatch: Sendable {
    internal let coordinator: MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>
    internal let streamRows: [MLXContinuousBatchStreamRow]

    internal init(
        coordinator: MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>,
        streamRows: [MLXContinuousBatchStreamRow]
    ) {
        self.coordinator = coordinator
        self.streamRows = streamRows
    }

    internal var orderedRowIDs: [MLXGenerationBatchRowID] {
        coordinator.orderedRowIDs
    }

    internal var count: Int {
        streamRows.count
    }

    internal var isEmpty: Bool {
        streamRows.isEmpty
    }
}
