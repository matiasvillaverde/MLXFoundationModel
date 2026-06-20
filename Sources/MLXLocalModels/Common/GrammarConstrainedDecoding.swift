import CXGrammarBridge
import Foundation
@preconcurrency import MLX

internal enum GrammarConstraintError: Error, LocalizedError, Sendable {
    case missingTokenizerJSON(URL)
    case invalidTokenizerJSON(String)
    case missingGrammarCompiler
    case bridge(String)
    case rejectedSampledToken(Int32)

    internal var errorDescription: String? {
        switch self {
        case .missingTokenizerJSON(let url):
            return "Missing tokenizer JSON at \(url.path)"
        case .invalidTokenizerJSON(let reason):
            return "Invalid tokenizer JSON: \(reason)"
        case .missingGrammarCompiler:
            return "Token-level grammar constraints are unavailable for this model"
        case .bridge(let message):
            return message
        case .rejectedSampledToken(let token):
            return "Grammar matcher rejected sampled token \(token)"
        }
    }
}

internal final class GrammarConstraintCompiler: @unchecked Sendable {
    private let handle: OpaquePointer
    private let vocabularySize: Int
    private let stopTokenIds: [Int32]
    private let stopTokenCount: Int

    internal init(modelDirectory: URL, stopTokenIds: Set<Int>) throws {
        let tokenizerURL = modelDirectory.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw GrammarConstraintError.missingTokenizerJSON(tokenizerURL)
        }
        let tokenizerJSON = try String(contentsOf: tokenizerURL, encoding: .utf8)
        let vocabulary = try TokenizerVocabularyParser.encodedVocabulary(from: tokenizerJSON)
        let stopTokens = stopTokenIds.map(Int32.init).sorted()
        self.vocabularySize = vocabulary.count
        self.stopTokenIds = stopTokens
        self.stopTokenCount = stopTokens.count

        self.handle = try Self.makeHandle(
            vocabulary: vocabulary,
            stopTokenIds: stopTokens,
            tokenizerJSON: tokenizerJSON
        )
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: .compilerReady,
            kind: nil,
            mode: nil,
            tokenCount: stopTokenCount,
            tokenID: nil,
            vocabularySize: vocabularySize,
            bitmaskSize: nil,
            isCompleted: nil,
            isTerminated: nil,
            message: nil
        ))
    }

    deinit {
        cxg_compiler_destroy(handle)
    }

    internal func makeMatcher(
        for configuration: GrammarSamplingConfiguration
    ) throws -> GrammarConstraintMatcher {
        let matcherHandle: OpaquePointer? = try withBridgeError { error in
            switch configuration.kind {
            case .builtinJSON:
                cxg_compiler_compile_builtin_json(handle, error)

            case .jsonSchema:
                cxg_compiler_compile_json_schema(
                    handle,
                    configuration.grammar,
                    configuration.strict,
                    error
                )

            case .ebnf, .choices:
                cxg_compiler_compile_ebnf(
                    handle,
                    configuration.grammar,
                    configuration.root,
                    error
                )

            case .structuralTag:
                cxg_compiler_compile_structural_tag(
                    handle,
                    configuration.grammar,
                    error
                )

            case .regex:
                cxg_compiler_compile_regex(handle, configuration.grammar, error)
            }
        }
        let matcher = GrammarConstraintMatcher(
            handle: try requireHandle(matcherHandle),
            kind: configuration.kind,
            stopTokenIds: stopTokenIds
        )
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: .matcherPrepared,
            kind: configuration.kind,
            mode: nil,
            tokenCount: nil,
            tokenID: nil,
            vocabularySize: matcher.vocabularySize,
            bitmaskSize: matcher.bitmaskSize,
            isCompleted: nil,
            isTerminated: nil,
            message: nil
        ))
        return matcher
    }

    private static func makeHandle(
        vocabulary: [String],
        stopTokenIds: [Int32],
        tokenizerJSON: String
    ) throws -> OpaquePointer {
        try vocabulary.withCStringArray { vocabPointer in
            try stopTokenIds.withUnsafeBufferPointer { stopPointer in
                let handle: OpaquePointer? = try withBridgeError { error in
                    cxg_compiler_create(
                        vocabPointer.baseAddress,
                        Int32(vocabulary.count),
                        stopPointer.baseAddress,
                        Int32(stopTokenIds.count),
                        tokenizerJSON,
                        error
                    )
                }
                return try requireHandle(handle)
            }
        }
    }
}

internal final class GrammarConstraintMatcher: @unchecked Sendable {
    private let handle: OpaquePointer
    internal let kind: GrammarConstraintKind
    internal let bitmaskSize: Int
    internal let vocabularySize: Int
    internal let stopTokenIds: [Int32]
    private var cachedMasks: [Data: GrammarTokenMask] = [:]

    internal init(handle: OpaquePointer, kind: GrammarConstraintKind, stopTokenIds: [Int32]) {
        self.handle = handle
        self.kind = kind
        self.stopTokenIds = stopTokenIds
        self.vocabularySize = Int(cxg_matcher_vocab_size(handle))
        self.bitmaskSize = Int(cxg_matcher_bitmask_size(handle))
    }

    deinit {
        cxg_matcher_destroy(handle)
    }

    internal func nextMask() throws -> GrammarTokenMask? {
        var bitmask = Array(repeating: Int32(-1), count: bitmaskSize)
        let shouldApply: Bool = try bitmask.withUnsafeMutableBufferPointer { pointer in
            try withBridgeError { error in
                cxg_matcher_fill_next_token_bitmask(
                    handle,
                    pointer.baseAddress,
                    Int32(bitmaskSize),
                    error
                )
            }
        }
        guard shouldApply else {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .maskSkipped,
                kind: kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: vocabularySize,
                bitmaskSize: bitmaskSize,
                isCompleted: isCompleted,
                isTerminated: isTerminated,
                message: "XGrammar reported no token mask for the current state"
            ))
            return nil
        }

        let key = bitmask.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        let mask: GrammarTokenMask
        if let cached = cachedMasks[key] {
            mask = cached
        } else {
            mask = GrammarTokenMask(bitmask: bitmask, vocabularySize: vocabularySize, cacheKey: key)
            cachedMasks[key] = mask
        }

        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: .maskApplied,
            kind: kind,
            mode: mask.mode,
            tokenCount: mask.tokenIDs.count,
            tokenID: nil,
            vocabularySize: vocabularySize,
            bitmaskSize: bitmaskSize,
            isCompleted: isCompleted,
            isTerminated: isTerminated,
            message: nil
        ))
        return mask
    }

    internal static func nextMasksBatched(
        for matchers: [GrammarConstraintMatcher]
    ) throws -> [GrammarTokenMask] {
        guard let first = matchers.first else {
            return []
        }
        try validateBatch(matchers, expected: first)

        var bitmasks = Array(
            repeating: Int32(-1),
            count: first.bitmaskSize * matchers.count
        )
        let handles: [OpaquePointer?] = matchers.map(\.handle)
        _ = try handles.withUnsafeBufferPointer { handlePointer in
            try bitmasks.withUnsafeMutableBufferPointer { bitmaskPointer in
                try withBridgeError { error in
                    cxg_matcher_batch_fill_next_token_bitmask(
                        handlePointer.baseAddress,
                        Int32(matchers.count),
                        bitmaskPointer.baseAddress,
                        Int32(matchers.count),
                        Int32(first.bitmaskSize),
                        error
                    )
                }
            }
        }

        var masks: [GrammarTokenMask] = []
        masks.reserveCapacity(matchers.count)
        for (index, matcher) in matchers.enumerated() {
            let lowerBound = index * first.bitmaskSize
            let upperBound = lowerBound + first.bitmaskSize
            let rowBitmask = Array(bitmasks[lowerBound ..< upperBound])
            let mask = matcher.cachedMask(for: rowBitmask)
            matcher.recordBatchMaskApplied(mask)
            masks.append(mask)
        }
        return masks
    }

    internal func accept(token: Int32) throws {
        do {
            let accepted: Bool = try withBridgeError { error in
                cxg_matcher_accept_token(handle, token, error)
            }
            guard accepted else {
                throw GrammarConstraintError.rejectedSampledToken(token)
            }
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .tokenAccepted,
                kind: kind,
                mode: nil,
                tokenCount: nil,
                tokenID: Int(token),
                vocabularySize: vocabularySize,
                bitmaskSize: bitmaskSize,
                isCompleted: isCompleted,
                isTerminated: isTerminated,
                message: nil
            ))
        } catch let error as GrammarConstraintError {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .tokenRejected,
                kind: kind,
                mode: nil,
                tokenCount: nil,
                tokenID: Int(token),
                vocabularySize: vocabularySize,
                bitmaskSize: bitmaskSize,
                isCompleted: isCompleted,
                isTerminated: isTerminated,
                message: error.localizedDescription
            ))
            throw error
        } catch {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .tokenRejected,
                kind: kind,
                mode: nil,
                tokenCount: nil,
                tokenID: Int(token),
                vocabularySize: vocabularySize,
                bitmaskSize: bitmaskSize,
                isCompleted: isCompleted,
                isTerminated: isTerminated,
                message: error.localizedDescription
            ))
            throw error
        }
    }

    internal var isCompleted: Bool {
        cxg_matcher_is_completed(handle)
    }

    internal var isTerminated: Bool {
        cxg_matcher_is_terminated(handle)
    }

    private static func validateBatch(
        _ matchers: [GrammarConstraintMatcher],
        expected: GrammarConstraintMatcher
    ) throws {
        for matcher in matchers {
            guard !matcher.isTerminated else {
                throw GrammarConstraintError.bridge(
                    "Cannot batch grammar masks with a terminated matcher"
                )
            }
            guard !matcher.isCompleted else {
                throw GrammarConstraintError.bridge(
                    "Cannot batch grammar masks with a completed matcher"
                )
            }
            guard matcher.vocabularySize == expected.vocabularySize &&
                matcher.bitmaskSize == expected.bitmaskSize
            else {
                throw GrammarConstraintError.bridge(
                    "Cannot batch grammar masks with different vocabularies"
                )
            }
        }
    }

    private func cachedMask(for bitmask: [Int32]) -> GrammarTokenMask {
        let key = bitmask.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        if let cached = cachedMasks[key] {
            return cached
        }
        let mask = GrammarTokenMask(bitmask: bitmask, vocabularySize: vocabularySize, cacheKey: key)
        cachedMasks[key] = mask
        return mask
    }

    private func recordBatchMaskApplied(_ mask: GrammarTokenMask) {
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: .batchMaskApplied,
            kind: kind,
            mode: mask.mode,
            tokenCount: mask.tokenIDs.count,
            tokenID: nil,
            vocabularySize: vocabularySize,
            bitmaskSize: bitmaskSize,
            isCompleted: isCompleted,
            isTerminated: isTerminated,
            message: nil
        ))
    }
}

internal struct GrammarTokenMask: Sendable {
    enum Mode: Sendable, Equatable {
        case allow
        case reject
    }

    let mode: Mode
    let tokenIDs: [Int32]
    let cacheKey: Data

    internal init(bitmask: [Int32], vocabularySize: Int, cacheKey: Data? = nil) {
        self.cacheKey = cacheKey ?? bitmask.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        let acceptedCount = Self.acceptedTokenCount(bitmask: bitmask, vocabularySize: vocabularySize)
        let rejectedCount = vocabularySize - acceptedCount
        let shouldBuildAccepted = acceptedCount <= rejectedCount
        if shouldBuildAccepted {
            self.mode = .allow
        } else {
            self.mode = .reject
        }
        self.tokenIDs = Self.tokenIDs(
            matchingAcceptedState: shouldBuildAccepted,
            bitmask: bitmask,
            vocabularySize: vocabularySize,
            capacity: min(acceptedCount, rejectedCount)
        )
    }

    internal init(mode: Mode, tokenIDs: [Int32], cacheKey: Data) {
        self.mode = mode
        self.tokenIDs = tokenIDs
        self.cacheKey = cacheKey
    }

    private static func tokenIDs(
        matchingAcceptedState shouldMatchAccepted: Bool,
        bitmask: [Int32],
        vocabularySize: Int,
        capacity: Int
    ) -> [Int32] {
        guard capacity > 0, vocabularySize > 0 else {
            return []
        }
        var result = Array(repeating: Int32(0), count: capacity)
        let written = bitmask.withUnsafeBufferPointer { pointer in
            result.withUnsafeMutableBufferPointer { output in
                cxg_bitmask_fill_token_ids(
                    pointer.baseAddress,
                    Int32(bitmask.count),
                    Int32(vocabularySize),
                    shouldMatchAccepted,
                    output.baseAddress,
                    Int32(output.count)
                )
            }
        }
        if written < result.count {
            result.removeSubrange(Int(written) ..< result.count)
        }
        return result
    }

    private static func acceptedTokenCount(bitmask: [Int32], vocabularySize: Int) -> Int {
        guard vocabularySize > 0 else {
            return 0
        }
        return bitmask.withUnsafeBufferPointer { pointer in
            Int(cxg_bitmask_count_accepted(
                pointer.baseAddress,
                Int32(bitmask.count),
                Int32(vocabularySize)
            ))
        }
    }
}

internal struct GrammarConstrainedLogitProcessor: LogitProcessor, @unchecked Sendable {
    private struct PreparedTokenMask {
        let mode: GrammarTokenMask.Mode
        let tokenCount: Int
        let indices: MLXArray
    }

    private static let preparedMaskCacheLimit = 256

    private let matcher: GrammarConstraintMatcher
    private let negInf = MLXArray(-Float.infinity)
    private var failure: GrammarConstraintError?
    private var preparedMasks: [Data: PreparedTokenMask] = [:]

    internal init(matcher: GrammarConstraintMatcher) {
        self.matcher = matcher
    }

    internal static func processBatched(
        logits: MLXArray,
        processors: inout [GrammarConstrainedLogitProcessor?]
    ) -> MLXArray {
        guard !processors.isEmpty else {
            return logits
        }
        precondition(logits.ndim == 2, "Batched grammar logits must be [batch, vocabulary]")
        precondition(
            logits.dim(0) == processors.count,
            "Batched grammar processor count must match logits row count"
        )

        var masked = logits
        var activeRows: [(index: Int, matcher: GrammarConstraintMatcher)] = []
        activeRows.reserveCapacity(processors.count)

        for index in processors.indices {
            guard var processor = processors[index] else {
                continue
            }
            switch processor.batchedMaskReadiness() {
            case .active:
                activeRows.append((index, processor.matcher))

            case .completed:
                masked = Self.replacingRow(index, in: masked) { row in
                    processor.processCompletedGrammar(logits: row)
                }

            case .failed:
                masked = Self.replacingRow(index, in: masked) { row in
                    processor.processFailedClosed(logits: row)
                }

            case .skipped:
                break
            }
            processors[index] = processor
        }

        guard !activeRows.isEmpty else {
            return masked
        }

        let tokenMasks: [GrammarTokenMask]
        do {
            tokenMasks = try GrammarConstraintMatcher.nextMasksBatched(
                for: activeRows.map(\.matcher)
            )
        } catch {
            return failClosedBatchedRows(
                error: error,
                rows: activeRows.map(\.index),
                logits: masked,
                processors: &processors
            )
        }

        for (activeRow, tokenMask) in zip(activeRows, tokenMasks) {
            guard var processor = processors[activeRow.index] else {
                continue
            }
            masked = Self.replacingRow(activeRow.index, in: masked) { row in
                processor.process(tokenMask: tokenMask, logits: row)
            }
            processors[activeRow.index] = processor
        }
        return masked
    }

    internal static func processBatched(
        logits: MLXArray,
        processorRows: inout MLXGenerationBatchRowTable<GrammarConstrainedLogitProcessor?>
    ) -> MLXArray {
        precondition(logits.ndim == 2, "Batched grammar logits must be [batch, vocabulary]")
        precondition(
            logits.dim(0) == processorRows.count,
            "Batched grammar processor row count must match logits row count"
        )

        var processors = processorRows.orderedPayloads
        let masked = processBatched(logits: logits, processors: &processors)
        do {
            try processorRows.replaceOrderedPayloads(processors)
        } catch {
            preconditionFailure("Batched grammar processor row table update failed: \(error)")
        }
        return masked
    }

    mutating internal func prompt(_ prompt: MLXArray) {}

    mutating internal func process(logits: MLXArray) -> MLXArray {
        guard !matcher.isTerminated else {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .maskSkipped,
                kind: matcher.kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: "Grammar matcher is terminated"
            ))
            return logits
        }
        guard !matcher.isCompleted else {
            return processCompletedGrammar(logits: logits)
        }
        if let failure {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .processorFailedClosed,
                kind: matcher.kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: failure.localizedDescription
            ))
            return MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)
        }
        let tokenMask: GrammarTokenMask?
        do {
            tokenMask = try matcher.nextMask()
        } catch let error as GrammarConstraintError {
            failure = error
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .processorFailedClosed,
                kind: matcher.kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: error.localizedDescription
            ))
            return processFailedClosed(logits: logits)
        } catch {
            let grammarError = GrammarConstraintError.bridge(error.localizedDescription)
            failure = grammarError
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .processorFailedClosed,
                kind: matcher.kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: grammarError.localizedDescription
            ))
            return processFailedClosed(logits: logits)
        }
        return process(tokenMask: tokenMask, logits: logits)
    }

    private enum BatchedMaskReadiness {
        case active
        case completed
        case failed
        case skipped
    }

    private mutating func batchedMaskReadiness() -> BatchedMaskReadiness {
        guard !matcher.isTerminated else {
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .maskSkipped,
                kind: matcher.kind,
                mode: nil,
                tokenCount: nil,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: "Grammar matcher is terminated"
            ))
            return .skipped
        }
        guard !matcher.isCompleted else {
            return .completed
        }
        guard failure == nil else {
            return .failed
        }
        return .active
    }

    private static func failClosedBatchedRows(
        error: Error,
        rows: [Int],
        logits: MLXArray,
        processors: inout [GrammarConstrainedLogitProcessor?]
    ) -> MLXArray {
        let grammarError = (error as? GrammarConstraintError) ?? .bridge(error.localizedDescription)
        var masked = logits
        for row in rows {
            guard var processor = processors[row] else {
                continue
            }
            processor.failure = grammarError
            processor.recordProcessorFailedClosed(grammarError)
            masked = replacingRow(row, in: masked) { rowLogits in
                processor.processFailedClosed(logits: rowLogits)
            }
            processors[row] = processor
        }
        return masked
    }

    private mutating func process(
        tokenMask: GrammarTokenMask?,
        logits: MLXArray
    ) -> MLXArray {
        guard let tokenMask else {
            return logits
        }
        guard !tokenMask.tokenIDs.isEmpty else {
            return processEmptyMask(tokenMask, logits: logits)
        }

        let preparedMask = preparedTokenMask(for: tokenMask)
        switch preparedMask.mode {
        case .allow:
            let masked = MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)
            masked[0..., preparedMask.indices] = logits[0..., preparedMask.indices]
            return masked

        case .reject:
            let masked = logits
            masked[0..., preparedMask.indices] = negInf
            return masked
        }
    }

    private func processFailedClosed(logits: MLXArray) -> MLXArray {
        MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)
    }

    private func recordProcessorFailedClosed(_ error: GrammarConstraintError) {
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: .processorFailedClosed,
            kind: matcher.kind,
            mode: nil,
            tokenCount: nil,
            tokenID: nil,
            vocabularySize: matcher.vocabularySize,
            bitmaskSize: matcher.bitmaskSize,
            isCompleted: matcher.isCompleted,
            isTerminated: matcher.isTerminated,
            message: error.localizedDescription
        ))
    }

    private static func replacingRow(
        _ rowIndex: Int,
        in logits: MLXArray,
        with transform: (MLXArray) -> MLXArray
    ) -> MLXArray {
        let rowRange = rowIndex ..< rowIndex + 1
        let row = logits[rowRange, 0...]
        let result = logits
        result[rowRange, 0...] = transform(row)
        return result
    }

    private mutating func processEmptyMask(_ tokenMask: GrammarTokenMask, logits: MLXArray) -> MLXArray {
        switch tokenMask.mode {
        case .allow:
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .processorFailedClosed,
                kind: matcher.kind,
                mode: tokenMask.mode,
                tokenCount: 0,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: "Grammar token mask rejected the full vocabulary"
            ))
            return MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)

        case .reject:
            MLXGenerationDiagnostics.recordGrammarConstraint(.init(
                stage: .maskSkipped,
                kind: matcher.kind,
                mode: tokenMask.mode,
                tokenCount: 0,
                tokenID: nil,
                vocabularySize: matcher.vocabularySize,
                bitmaskSize: matcher.bitmaskSize,
                isCompleted: matcher.isCompleted,
                isTerminated: matcher.isTerminated,
                message: "Grammar token mask accepts the full vocabulary"
            ))
            return logits
        }
    }

    private mutating func processCompletedGrammar(logits: MLXArray) -> MLXArray {
        let vocabularyLimit = logits.shape.last ?? matcher.vocabularySize
        let stopTokenIds = matcher.stopTokenIds
            .filter { tokenID in
                tokenID >= 0 && Int(tokenID) < vocabularyLimit
            }
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: stopTokenIds.isEmpty ? .maskSkipped : .maskApplied,
            kind: matcher.kind,
            mode: stopTokenIds.isEmpty ? nil : .allow,
            tokenCount: stopTokenIds.isEmpty ? nil : stopTokenIds.count,
            tokenID: nil,
            vocabularySize: matcher.vocabularySize,
            bitmaskSize: matcher.bitmaskSize,
            isCompleted: matcher.isCompleted,
            isTerminated: matcher.isTerminated,
            message: stopTokenIds.isEmpty
                ? "Grammar matcher is completed with no stop tokens available"
                : "Grammar matcher is completed; forcing stop tokens"
        ))
        guard !stopTokenIds.isEmpty else {
            return logits
        }
        let key = stopTokenIds.withUnsafeBufferPointer { pointer in
            Data(buffer: pointer)
        }
        let tokenMask = GrammarTokenMask(mode: .allow, tokenIDs: stopTokenIds, cacheKey: key)
        let preparedMask = preparedTokenMask(for: tokenMask)
        let masked = MLXArray.full(logits.shape, values: negInf, dtype: logits.dtype)
        masked[0..., preparedMask.indices] = logits[0..., preparedMask.indices]
        return masked
    }

    private mutating func preparedTokenMask(for tokenMask: GrammarTokenMask) -> PreparedTokenMask {
        if let prepared = preparedMasks[tokenMask.cacheKey] {
            recordPreparedMask(stage: .mlxMaskReused, tokenMask: tokenMask)
            return prepared
        }
        if preparedMasks.count >= Self.preparedMaskCacheLimit {
            preparedMasks.removeAll(keepingCapacity: true)
        }
        let prepared = PreparedTokenMask(
            mode: tokenMask.mode,
            tokenCount: tokenMask.tokenIDs.count,
            indices: MLXArray(tokenMask.tokenIDs).asType(.int32)
        )
        preparedMasks[tokenMask.cacheKey] = prepared
        recordPreparedMask(stage: .mlxMaskPrepared, tokenMask: tokenMask)
        return prepared
    }

    private func recordPreparedMask(
        stage: MLXGrammarConstraintSnapshot.Stage,
        tokenMask: GrammarTokenMask
    ) {
        MLXGenerationDiagnostics.recordGrammarConstraint(.init(
            stage: stage,
            kind: matcher.kind,
            mode: tokenMask.mode,
            tokenCount: tokenMask.tokenIDs.count,
            tokenID: nil,
            vocabularySize: matcher.vocabularySize,
            bitmaskSize: matcher.bitmaskSize,
            isCompleted: matcher.isCompleted,
            isTerminated: matcher.isTerminated,
            message: nil
        ))
    }

    mutating internal func didSample(token: MLXArray) {
        guard !matcher.isTerminated, !matcher.isCompleted else {
            return
        }
        let tokenID = Int32(token.item(Int.self))
        do {
            try matcher.accept(token: tokenID)
        } catch let error as GrammarConstraintError {
            failure = error
        } catch {
            failure = .bridge(error.localizedDescription)
        }
    }
}

private enum TokenizerVocabularyParser {
    static func encodedVocabulary(from tokenizerJSON: String) throws -> [String] {
        guard let data = tokenizerJSON.data(using: .utf8) else {
            throw GrammarConstraintError.invalidTokenizerJSON("Tokenizer JSON is not UTF-8")
        }

        let bytes = Array(data)
        var vocabularyParser = TokenizerJSONVocabularyParser(bytes: bytes)
        var entries = try vocabularyParser.parseVocabularyEntries()
        var addedTokenParser = TokenizerJSONVocabularyParser(bytes: bytes)
        entries.merge(try addedTokenParser.parseAddedTokenEntries()) { _, new in new }
        return try compactVocabulary(from: entries)
    }

    private static func compactVocabulary(from entries: [Int: String]) throws -> [String] {
        guard let maxID = entries.keys.max(), maxID >= 0 else {
            throw GrammarConstraintError.invalidTokenizerJSON("Vocabulary is empty")
        }
        var vocabulary = Array(repeating: "", count: maxID + 1)
        var present = Array(repeating: false, count: maxID + 1)
        for (id, token) in entries {
            guard id >= 0 else {
                throw GrammarConstraintError.invalidTokenizerJSON("Vocabulary contains negative token id")
            }
            vocabulary[id] = token
            present[id] = true
        }
        if let missingID = present.firstIndex(of: false) {
            throw GrammarConstraintError.invalidTokenizerJSON("Vocabulary is missing token id \(missingID)")
        }
        return vocabulary
    }
}

private struct TokenizerJSONVocabularyParser {
    private static let vocabKey = Array(#""vocab""#.utf8)
    private static let addedTokensKey = Array(#""added_tokens""#.utf8)

    var bytes: [UInt8]
    var index = 0

    mutating func parseVocabularyEntries() throws -> [Int: String] {
        try seekKey(Self.vocabKey, missingMessage: "Missing model.vocab")
        if try consumeIfPresent(byte: UInt8(ascii: "{")) {
            return try parseObjectEntries()
        }
        if try consumeIfPresent(byte: UInt8(ascii: "[")) {
            return try parseArrayEntries()
        }
        throw GrammarConstraintError.invalidTokenizerJSON("Unsupported vocabulary shape")
    }

    mutating func parseAddedTokenEntries() throws -> [Int: String] {
        guard try seekOptionalKey(Self.addedTokensKey) else {
            return [:]
        }
        try consume(byte: UInt8(ascii: "["))

        var entries: [Int: String] = [:]
        try skipWhitespace()
        while try !consumeIfPresent(byte: UInt8(ascii: "]")) {
            let token = try parseAddedTokenObject()
            if let id = token.id, let content = token.content {
                entries[id] = content
            }
            try skipWhitespace()
            if try consumeIfPresent(byte: UInt8(ascii: ",")) {
                try skipWhitespace()
            } else {
                try consume(byte: UInt8(ascii: "]"))
                break
            }
        }
        return entries
    }

    private mutating func seekKey(_ key: [UInt8], missingMessage: String) throws {
        guard try seekOptionalKey(key) else {
            throw GrammarConstraintError.invalidTokenizerJSON(missingMessage)
        }
    }

    private mutating func seekOptionalKey(_ key: [UInt8]) throws -> Bool {
        while index <= bytes.count - key.count {
            if matches(key) {
                index += key.count
                try skipWhitespace()
                try consume(byte: UInt8(ascii: ":"))
                try skipWhitespace()
                return true
            }
            index += 1
        }
        return false
    }

    private func matches(_ key: [UInt8]) -> Bool {
        guard index + key.count <= bytes.count else {
            return false
        }
        for offset in 0 ..< key.count where bytes[index + offset] != key[offset] {
            return false
        }
        return true
    }

    private mutating func parseObjectEntries() throws -> [Int: String] {
        var entries: [Int: String] = [:]
        try skipWhitespace()
        while try !consumeIfPresent(byte: UInt8(ascii: "}")) {
            let token = try parseString()
            try skipWhitespace()
            try consume(byte: UInt8(ascii: ":"))
            try skipWhitespace()
            let tokenID = try parseInteger()
            entries[tokenID] = token
            try skipWhitespace()
            if try consumeIfPresent(byte: UInt8(ascii: ",")) {
                try skipWhitespace()
            } else {
                try consume(byte: UInt8(ascii: "}"))
                break
            }
        }
        return entries
    }

    private mutating func parseArrayEntries() throws -> [Int: String] {
        var entries: [Int: String] = [:]
        try skipWhitespace()
        while try !consumeIfPresent(byte: UInt8(ascii: "]")) {
            try consume(byte: UInt8(ascii: "["))
            try skipWhitespace()
            let token = try parseString()
            entries[entries.count] = token
            try skipWhitespace()
            while try consumeIfPresent(byte: UInt8(ascii: ",")) {
                try skipValue()
                try skipWhitespace()
            }
            try consume(byte: UInt8(ascii: "]"))
            try skipWhitespace()
            if try consumeIfPresent(byte: UInt8(ascii: ",")) {
                try skipWhitespace()
            } else {
                try consume(byte: UInt8(ascii: "]"))
                break
            }
        }
        return entries
    }

    private mutating func parseAddedTokenObject() throws -> (id: Int?, content: String?) {
        try consume(byte: UInt8(ascii: "{"))
        var id: Int?
        var content: String?

        try skipWhitespace()
        while try !consumeIfPresent(byte: UInt8(ascii: "}")) {
            let key = try parseString()
            try skipWhitespace()
            try consume(byte: UInt8(ascii: ":"))
            try skipWhitespace()
            switch key {
            case "id":
                id = try parseInteger()

            case "content":
                content = try parseString()

            default:
                try skipValue()
            }
            try skipWhitespace()
            if try consumeIfPresent(byte: UInt8(ascii: ",")) {
                try skipWhitespace()
            } else {
                try consume(byte: UInt8(ascii: "}"))
                break
            }
        }

        return (id, content)
    }

    private mutating func parseString() throws -> String {
        try consume(byte: UInt8(ascii: "\""))
        var output: [UInt8] = []
        while index < bytes.count {
            let byte = bytes[index]
            index += 1
            if byte == UInt8(ascii: "\"") {
                return String(decoding: output, as: UTF8.self)
            }
            if byte == UInt8(ascii: "\\" ) {
                try appendEscapedStringByte(to: &output)
            } else {
                output.append(byte)
            }
        }
        throw GrammarConstraintError.invalidTokenizerJSON("Unterminated JSON string in vocab")
    }

    private mutating func appendEscapedStringByte(to output: inout [UInt8]) throws {
        guard index < bytes.count else {
            throw GrammarConstraintError.invalidTokenizerJSON("Unterminated JSON string escape")
        }
        let escaped = bytes[index]
        index += 1
        switch escaped {
        case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"):
            output.append(escaped)
        case UInt8(ascii: "b"):
            output.append(8)
        case UInt8(ascii: "f"):
            output.append(12)
        case UInt8(ascii: "n"):
            output.append(10)
        case UInt8(ascii: "r"):
            output.append(13)
        case UInt8(ascii: "t"):
            output.append(9)
        case UInt8(ascii: "u"):
            try appendUnicodeEscape(to: &output)
        default:
            throw GrammarConstraintError.invalidTokenizerJSON("Invalid JSON string escape")
        }
    }

    private mutating func appendUnicodeEscape(to output: inout [UInt8]) throws {
        let scalarValue = try parseHexScalar()
        if (0xD800 ... 0xDBFF).contains(scalarValue) {
            try consume(byte: UInt8(ascii: "\\"))
            try consume(byte: UInt8(ascii: "u"))
            let lowSurrogate = try parseHexScalar()
            guard (0xDC00 ... 0xDFFF).contains(lowSurrogate) else {
                throw GrammarConstraintError.invalidTokenizerJSON("Invalid JSON unicode surrogate")
            }
            let high = scalarValue - 0xD800
            let low = lowSurrogate - 0xDC00
            try appendScalar(0x10000 + ((high << 10) | low), to: &output)
        } else {
            try appendScalar(scalarValue, to: &output)
        }
    }

    private mutating func parseHexScalar() throws -> UInt32 {
        guard index + 4 <= bytes.count else {
            throw GrammarConstraintError.invalidTokenizerJSON("Incomplete JSON unicode escape")
        }
        var value: UInt32 = 0
        for _ in 0 ..< 4 {
            value = (value << 4) | UInt32(try hexValue(bytes[index]))
            index += 1
        }
        return value
    }

    private func appendScalar(_ value: UInt32, to output: inout [UInt8]) throws {
        guard let scalar = UnicodeScalar(value) else {
            throw GrammarConstraintError.invalidTokenizerJSON("Invalid JSON unicode scalar")
        }
        output.append(contentsOf: String(scalar).utf8)
    }

    private func hexValue(_ byte: UInt8) throws -> UInt8 {
        switch byte {
        case UInt8(ascii: "0") ... UInt8(ascii: "9"):
            return byte - UInt8(ascii: "0")
        case UInt8(ascii: "a") ... UInt8(ascii: "f"):
            return byte - UInt8(ascii: "a") + 10
        case UInt8(ascii: "A") ... UInt8(ascii: "F"):
            return byte - UInt8(ascii: "A") + 10
        default:
            throw GrammarConstraintError.invalidTokenizerJSON("Invalid JSON unicode escape")
        }
    }

    private mutating func parseInteger() throws -> Int {
        var sign = 1
        if try consumeIfPresent(byte: UInt8(ascii: "-")) {
            sign = -1
        }
        var value = 0
        var consumedDigit = false
        while index < bytes.count, (UInt8(ascii: "0") ... UInt8(ascii: "9")).contains(bytes[index]) {
            consumedDigit = true
            value = (value * 10) + Int(bytes[index] - UInt8(ascii: "0"))
            index += 1
        }
        guard consumedDigit else {
            throw GrammarConstraintError.invalidTokenizerJSON("Expected integer vocab id")
        }
        return value * sign
    }

    private mutating func skipValue() throws {
        try skipWhitespace()
        guard index < bytes.count else {
            throw GrammarConstraintError.invalidTokenizerJSON("Unexpected end of JSON value")
        }
        switch bytes[index] {
        case UInt8(ascii: "\""):
            _ = try parseString()

        case UInt8(ascii: "{"), UInt8(ascii: "["):
            try skipCompositeValue()

        case UInt8(ascii: "t"):
            try consumeLiteral("true")

        case UInt8(ascii: "f"):
            try consumeLiteral("false")

        case UInt8(ascii: "n"):
            try consumeLiteral("null")

        default:
            try skipNumber()
        }
    }

    private mutating func skipCompositeValue() throws {
        var stack: [UInt8] = []
        if try consumeIfPresent(byte: UInt8(ascii: "{")) {
            stack.append(UInt8(ascii: "}"))
        } else {
            try consume(byte: UInt8(ascii: "["))
            stack.append(UInt8(ascii: "]"))
        }

        while let expected = stack.last {
            guard index < bytes.count else {
                throw GrammarConstraintError.invalidTokenizerJSON("Unterminated JSON value")
            }
            switch bytes[index] {
            case UInt8(ascii: "\""):
                _ = try parseString()

            case UInt8(ascii: "{"):
                index += 1
                stack.append(UInt8(ascii: "}"))

            case UInt8(ascii: "["):
                index += 1
                stack.append(UInt8(ascii: "]"))

            case expected:
                index += 1
                stack.removeLast()

            default:
                index += 1
            }
        }
    }

    private mutating func consumeLiteral(_ literal: String) throws {
        for byte in literal.utf8 {
            try consume(byte: byte)
        }
    }

    private mutating func skipNumber() throws {
        var consumed = false
        while index < bytes.count {
            switch bytes[index] {
            case UInt8(ascii: "0") ... UInt8(ascii: "9"),
                UInt8(ascii: "-"),
                UInt8(ascii: "+"),
                UInt8(ascii: "."),
                UInt8(ascii: "e"),
                UInt8(ascii: "E"):
                consumed = true
                index += 1

            default:
                guard consumed else {
                    throw GrammarConstraintError.invalidTokenizerJSON("Unexpected JSON value")
                }
                return
            }
        }
        guard consumed else {
            throw GrammarConstraintError.invalidTokenizerJSON("Unexpected JSON value")
        }
    }

    private mutating func skipWhitespace() throws {
        while index < bytes.count {
            switch bytes[index] {
            case UInt8(ascii: " "), UInt8(ascii: "\n"), UInt8(ascii: "\r"), UInt8(ascii: "\t"):
                index += 1
            default:
                return
            }
        }
    }

    private mutating func consume(byte: UInt8) throws {
        guard try consumeIfPresent(byte: byte) else {
            throw GrammarConstraintError.invalidTokenizerJSON("Unexpected JSON token in vocab")
        }
    }

    private mutating func consumeIfPresent(byte: UInt8) throws -> Bool {
        guard index < bytes.count else {
            return false
        }
        guard bytes[index] == byte else {
            return false
        }
        index += 1
        return true
    }
}

private func requireHandle(_ handle: OpaquePointer?) throws -> OpaquePointer {
    guard let handle else {
        throw GrammarConstraintError.bridge("XGrammar returned a null handle")
    }
    return handle
}

private func withBridgeError<Result>(
    _ operation: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Result
) throws -> Result {
    var errorMessage: UnsafeMutablePointer<CChar>?
    let result = operation(&errorMessage)
    if let errorMessage {
        let message = String(cString: errorMessage)
        cxg_free_string(errorMessage)
        throw GrammarConstraintError.bridge(message)
    }
    return result
}

private extension Array where Element == String {
    func withCStringArray<Result>(
        _ body: (UnsafeBufferPointer<UnsafePointer<CChar>?>) throws -> Result
    ) throws -> Result {
        let cStrings = map { strdup($0) }
        defer {
            for string in cStrings {
                free(string)
            }
        }
        let pointers = cStrings.map { UnsafePointer<CChar>($0) }
        return try pointers.withUnsafeBufferPointer(body)
    }
}
