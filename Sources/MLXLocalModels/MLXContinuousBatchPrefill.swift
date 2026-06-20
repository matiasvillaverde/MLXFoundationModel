import MLX

internal enum MLXContinuousBatchPrefill {
    internal static func run(
        model: any LanguageModel,
        requests: [MLXContinuousBatchPrefillRequest],
        grammarCompiler: GrammarConstraintCompiler? = nil
    ) throws -> MLXContinuousBatchPrefillResult {
        try validate(requests)

        let parameters = requests[0].parameters
        let cache = try MLXContinuousBatchPrefixCacheMerger.initialCache(
            model: model,
            parameters: parameters,
            requests: requests
        )
        var logitRows = try makeLogitRows(
            requests: requests,
            grammarCompiler: grammarCompiler
        )
        let logits = model(
            LMInput.Text(tokens: tokenMatrix(for: requests)),
            cache: cache.isEmpty ? nil : cache,
            state: nil
        ).logits[0..., -1, 0...]
        let firstTokens = try logitRows.sample(logits: logits)
        let tokenIDs = firstTokens.tokenIDs
        let active = try activeRowsAfterFirstToken(
            tokenIDs,
            requests: requests,
            logitRows: logitRows
        )
        let batch = try generationBatch(for: active.requests)
        return MLXContinuousBatchPrefillResult(
            batch: batch,
            cache: cache,
            firstTokenIDs: tokenIDs,
            logitRows: active.logitRows
        )
    }

    private static func validate(
        _ requests: [MLXContinuousBatchPrefillRequest]
    ) throws {
        guard let first = requests.first else {
            throw MLXContinuousBatchPrefillError.emptyRequests
        }
        guard !first.promptTokenIDs.isEmpty else {
            throw MLXContinuousBatchPrefillError.emptyPrompt(rowIndex: 0)
        }

        for index in requests.indices {
            let request = requests[index]
            if request.promptTokenIDs.isEmpty {
                throw MLXContinuousBatchPrefillError.emptyPrompt(rowIndex: index)
            }
            if request.promptTokenIDs.count != first.promptTokenIDs.count {
                throw MLXContinuousBatchPrefillError.mismatchedPromptTokenCount(
                    expected: first.promptTokenIDs.count,
                    actual: request.promptTokenIDs.count,
                    rowIndex: index
                )
            }
            guard MLXContinuousBatchCacheSignature(parameters: request.parameters)
                == MLXContinuousBatchCacheSignature(parameters: first.parameters)
            else {
                throw MLXContinuousBatchPrefillError.incompatibleCacheParameters(rowIndex: index)
            }
            guard request.prefixCacheGroupKey == first.prefixCacheGroupKey else {
                throw MLXContinuousBatchPrefillError.incompatiblePrefixCache(rowIndex: index)
            }
        }
    }

    private static func makeLogitRows(
        requests: [MLXContinuousBatchPrefillRequest],
        grammarCompiler: GrammarConstraintCompiler?
    ) throws -> MLXContinuousBatchLogitRows {
        var rows = MLXContinuousBatchLogitRows()
        for index in requests.indices {
            try rows.append(
                id: MLXGenerationBatchRowID(index),
                row: .init(
                    processor: try requests[index].processor(
                        grammarCompiler: grammarCompiler
                    ),
                    sampler: requests[index].parameters.sampler()
                )
            )
        }
        return rows
    }

    private static func tokenMatrix(
        for requests: [MLXContinuousBatchPrefillRequest]
    ) -> MLXArray {
        let width = requests[0].promptTokenIDs.count
        let tokenIDs = requests.flatMap(\.promptTokenIDs)
        return MLXArray(tokenIDs).reshaped([requests.count, width])
    }

    private static func activeRowsAfterFirstToken(
        _ tokenIDs: [Int],
        requests: [MLXContinuousBatchPrefillRequest],
        logitRows: MLXContinuousBatchLogitRows
    ) throws -> (
        requests: [MLXContinuousBatchGenerationRequest],
        logitRows: MLXContinuousBatchLogitRows
    ) {
        var activeRows = ActivePrefillRows(capacity: requests.count)

        for index in tokenIDs.indices {
            var row = MLXContinuousBatchGenerationRow(
                previousTokenID: tokenIDs[index],
                maximumTokenCount: requests[index].parameters.maxTokens ?? Int.max,
                stopTokenIDs: requests[index].stopTokenIDs
            )
            let modelFinishReason = row.accept(tokenID: tokenIDs[index])
            let streamFinishReason = streamFirstTokenIfNeeded(
                tokenIDs[index],
                request: requests[index],
                modelFinishReason: modelFinishReason
            )
            if let finishReason = streamFinishReason ?? modelFinishReason {
                requests[index].sink.finish(finishReason)
                continue
            }
            try activeRows.append(
                request: requests[index],
                row: row,
                sourceRowID: MLXGenerationBatchRowID(index),
                sourceLogitRows: logitRows
            )
        }
        return (activeRows.requests, activeRows.logitRows)
    }

    private static func streamFirstTokenIfNeeded(
        _ tokenID: Int,
        request: MLXContinuousBatchPrefillRequest,
        modelFinishReason: MLXContinuousBatchFinishReason?
    ) -> MLXContinuousBatchFinishReason? {
        if case .stopToken = modelFinishReason {
            return nil
        }
        switch request.stream(tokenID: tokenID) {
        case .finish(let reason):
            return reason

        case .streamed, .suppressed:
            return nil
        }
    }

    private static func emptyBatch() throws -> MLXContinuousBatchGenerationBatch {
        MLXContinuousBatchGenerationBatch(
            coordinator: MLXContinuousBatchCoordinator(),
            streamRows: []
        )
    }

    private static func generationBatch(
        for requests: [MLXContinuousBatchGenerationRequest]
    ) throws -> MLXContinuousBatchGenerationBatch {
        try requests.isEmpty
            ? emptyBatch()
            : MLXContinuousBatchAssembler.assemble(requests: requests)
    }

    private struct ActivePrefillRows {
        private(set) var requests: [MLXContinuousBatchGenerationRequest]
        private(set) var logitRows = MLXContinuousBatchLogitRows()

        init(capacity: Int) {
            self.requests = []
            self.requests.reserveCapacity(capacity)
        }

        mutating func append(
            request: MLXContinuousBatchPrefillRequest,
            row: MLXContinuousBatchGenerationRow,
            sourceRowID: MLXGenerationBatchRowID,
            sourceLogitRows: MLXContinuousBatchLogitRows
        ) throws {
            if let sourceLogitRow = sourceLogitRows[sourceRowID] {
                try logitRows.append(
                    id: MLXGenerationBatchRowID(requests.count),
                    row: sourceLogitRow
                )
            }
            requests.append(request.continuousBatchGenerationRequest(
                previousTokenID: row.previousTokenID,
                generatedTokenCount: row.generatedTokenCount
            ))
        }
    }
}
