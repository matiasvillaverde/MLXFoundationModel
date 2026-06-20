internal enum MLXContinuousBatchAssembler {
    internal static func assemble(
        requests: [MLXContinuousBatchGenerationRequest],
        pagedKVBlockCapacity: Int = 0
    ) throws -> MLXContinuousBatchGenerationBatch {
        guard !requests.isEmpty else {
            throw MLXContinuousBatchCoordinatorError.emptyAdmission
        }

        var coordinator = MLXContinuousBatchCoordinator<MLXContinuousBatchGenerationRow>(
            pagedKVBlockCapacity: pagedKVBlockCapacity
        )
        let rowIDs = try coordinator.admitBatch(requests.map(\.generationRow))
        let streamRows = zip(rowIDs, requests).map { rowID, request in
            MLXContinuousBatchStreamRow(
                id: rowID,
                sink: request.sink,
                handleToken: request.handleToken
            )
        }

        return MLXContinuousBatchGenerationBatch(
            coordinator: coordinator,
            streamRows: streamRows
        )
    }
}
