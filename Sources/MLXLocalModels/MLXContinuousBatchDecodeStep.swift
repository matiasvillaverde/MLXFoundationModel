import MLX

internal struct MLXContinuousBatchDecodeStep: MLXContinuousBatchDecodingStrategy {
    internal let model: any LanguageModel
    internal var cache: [KVCache]
    internal var state: LMOutput.State?
    internal var logitRows: MLXContinuousBatchLogitRows
    internal let kvBits: Int?
    internal let kvGroupSize: Int
    internal let quantizedKVStart: Int
    internal let quantizedKVSkipLastLayer: Bool

    internal init(
        model: any LanguageModel,
        cache: [KVCache],
        logitRows: MLXContinuousBatchLogitRows,
        state: LMOutput.State? = nil,
        kvBits: Int? = nil,
        kvGroupSize: Int = GenerationConstants.defaultKVCacheGroupSize,
        quantizedKVStart: Int = 0,
        quantizedKVSkipLastLayer: Bool = false
    ) {
        self.model = model
        self.cache = cache
        self.state = state
        self.logitRows = logitRows
        self.kvBits = kvBits
        self.kvGroupSize = kvGroupSize
        self.quantizedKVStart = quantizedKVStart
        self.quantizedKVSkipLastLayer = quantizedKVSkipLastLayer
    }

    internal var rowCount: Int {
        logitRows.count
    }

    internal var orderedRowIDs: [MLXGenerationBatchRowID] {
        logitRows.orderedRowIDs
    }

    internal mutating func step(
        previousTokens: MLXArray
    ) throws -> MLXContinuousBatchSampledTokens {
        try validate(previousTokens: previousTokens)
        try validateCacheRowCount()

        let result = model(
            .init(tokens: tokenMatrix(from: previousTokens)),
            cache: cache.isEmpty ? nil : cache,
            state: state
        )
        state = result.state
        maybeQuantizeKVCache(
            cache: &cache,
            kvBits: kvBits,
            kvGroupSize: kvGroupSize,
            quantizedKVStart: quantizedKVStart,
            skipLastLayer: quantizedKVSkipLastLayer
        )
        MLXGenerationDiagnostics.recordCacheSnapshot(label: "continuous-batch-step", cache: cache)

        return try logitRows.sample(logits: result.logits[0..., -1, 0...])
    }

    internal mutating func realign(to orderedRowIDs: [MLXGenerationBatchRowID]) throws {
        let previousRowIDs = self.orderedRowIDs
        try logitRows.keep(ids: orderedRowIDs)
        try keepCacheRows(orderedRowIDs, from: previousRowIDs)
        try validateCacheRowCount()
    }

    private func validate(previousTokens: MLXArray) throws {
        guard rowCount > 0 else {
            throw MLXContinuousBatchDecodeStepError.emptyRows
        }
        switch previousTokens.ndim {
        case 1:
            try validateRowCount(previousTokens.dim(0))

        case 2:
            try validateRowCount(previousTokens.dim(0))
            guard previousTokens.dim(1) == 1 else {
                throw MLXContinuousBatchDecodeStepError.invalidTokenColumnCount(previousTokens.dim(1))
            }

        default:
            throw MLXContinuousBatchDecodeStepError.invalidTokenRank(previousTokens.ndim)
        }
    }

    private func validateRowCount(_ actual: Int) throws {
        guard actual == rowCount else {
            throw MLXContinuousBatchDecodeStepError.rowCountMismatch(
                expected: rowCount,
                actual: actual
            )
        }
    }

    private func validateCacheRowCount() throws {
        for layer in cache {
            guard let firstState = layer.state.first, firstState.ndim > 0 else {
                continue
            }
            let actual = firstState.dim(0)
            guard actual == rowCount else {
                throw MLXContinuousBatchDecodeStepError.cacheRowCountMismatch(
                    expected: rowCount,
                    actual: actual
                )
            }
        }
    }

    private func tokenMatrix(from previousTokens: MLXArray) -> MLXArray {
        previousTokens.ndim == 1
            ? previousTokens.reshaped([previousTokens.dim(0), 1])
            : previousTokens
    }

    private mutating func keepCacheRows(
        _ orderedRowIDs: [MLXGenerationBatchRowID],
        from previousRowIDs: [MLXGenerationBatchRowID]
    ) throws {
        guard orderedRowIDs != previousRowIDs else {
            return
        }

        let keepIndices = try cacheKeepIndices(
            orderedRowIDs,
            previousRowIDs: previousRowIDs
        )
        let indexArray = MLXArray(keepIndices)

        for index in cache.indices {
            let state = cache[index].state
            guard !state.isEmpty else {
                continue
            }

            var didFilter = false
            let filteredState = state.map { array in
                guard array.ndim > 0, array.dim(0) == previousRowIDs.count else {
                    return array
                }
                didFilter = true
                return array[indexArray]
            }

            if didFilter {
                cache[index].state = filteredState
            }
        }
    }

    private func cacheKeepIndices(
        _ orderedRowIDs: [MLXGenerationBatchRowID],
        previousRowIDs: [MLXGenerationBatchRowID]
    ) throws -> [Int] {
        var indexByID: [MLXGenerationBatchRowID: Int] = [:]
        indexByID.reserveCapacity(previousRowIDs.count)
        for (index, id) in previousRowIDs.enumerated() {
            indexByID[id] = index
        }

        return try orderedRowIDs.map { id in
            guard let index = indexByID[id] else {
                throw MLXGenerationBatchRowTableError.missingRowID(id)
            }
            return index
        }
    }
}
