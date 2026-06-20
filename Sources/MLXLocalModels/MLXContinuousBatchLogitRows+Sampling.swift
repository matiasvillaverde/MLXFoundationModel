import MLX

extension MLXContinuousBatchLogitRows {
    internal mutating func sample(
        logits: MLXArray
    ) throws -> MLXContinuousBatchSampledTokens {
        try validate(logits: logits)

        var payloads = rows.orderedPayloads
        var tokenArrays: [MLXArray] = []
        tokenArrays.reserveCapacity(payloads.count)

        for index in payloads.indices {
            let rowLogits = logits[index ..< index + 1, 0...]
            tokenArrays.append(payloads[index].sample(logits: rowLogits).reshaped([1]))
        }

        try rows.replaceOrderedPayloads(payloads)

        let tokenArray = concatenated(tokenArrays)
        eval(tokenArray)
        let tokenIDs = tokenArray.asArray(Int.self)
        recordSampled(tokenIDs: tokenIDs)
        return MLXContinuousBatchSampledTokens(
            rowIDs: rows.orderedIDs,
            tokenIDs: tokenIDs,
            tokenArray: tokenArray
        )
    }

    private func validate(logits: MLXArray) throws {
        guard logits.ndim == 2 else {
            throw MLXContinuousBatchLogitRowsError.invalidLogitRank(logits.ndim)
        }
        guard logits.dim(0) == rows.count else {
            throw MLXContinuousBatchLogitRowsError.rowCountMismatch(
                expected: rows.count,
                actual: logits.dim(0)
            )
        }
    }

    private func recordSampled(tokenIDs: [Int]) {
        MLXGenerationDiagnostics.recordContinuousBatchLogits(.init(
            stage: .sampled,
            rowCount: rows.count,
            rowIDs: rows.orderedIDs.map(\.rawValue),
            tokenIDs: tokenIDs
        ))
    }
}
