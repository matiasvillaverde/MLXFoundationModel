// Copyright © 2024 Apple Inc.

import MLX

import Tokenizers

/// Marker protocol for LLMModels
internal protocol LLMModel: LanguageModel, LoRAModel {
}

extension LLMModel {
    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small amount of
    /// tokens left to feed into the `TokenIterator`.
    internal func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult {
        let prefillStepSize = windowSize ?? GenerationConstants.defaultPrefillStepSize
        let controller = MLXGenerationDiagnostics.currentAdaptivePrefillController
        var y = input.text

        // prepare the prompt in chunks if larger than the prefill size
        while true {
            let remainingTokenCount = y.tokens.size
            let chunkSize = adaptiveChunkSize(
                remainingTokenCount: remainingTokenCount,
                prefillStepSize: prefillStepSize,
                controller: controller
            )
            guard remainingTokenCount > chunkSize else {
                break
            }
            let memoryBeforeBytes = Int64(Memory.activeMemory)
            let input = y[.newAxis, ..<chunkSize]
            _ = self(input, cache: cache.isEmpty ? nil : cache, state: nil)
            eval(cache)
            let memoryAfterBytes = Int64(Memory.activeMemory)
            controller?.recordChunk(
                tokenCount: chunkSize,
                memoryBeforeBytes: memoryBeforeBytes,
                memoryAfterBytes: memoryAfterBytes
            )
            MLXGenerationDiagnostics.recordPrefillChunk(
                chunkSize: chunkSize,
                remainingTokenCount: remainingTokenCount,
                prefillStepSize: prefillStepSize,
                memoryBeforeBytes: memoryBeforeBytes,
                memoryAfterBytes: memoryAfterBytes
            )
            y = y[chunkSize...]
        }

        return .tokens(y)
    }

    private func adaptiveChunkSize(
        remainingTokenCount: Int,
        prefillStepSize: Int,
        controller: MLXAdaptivePrefillChunkController?
    ) -> Int {
        let requested = min(max(1, prefillStepSize), max(1, remainingTokenCount))
        guard let controller else {
            return requested
        }
        let decision = controller.decision(remainingTokenCount: remainingTokenCount)
        MLXGenerationDiagnostics.recordAdaptivePrefillChunk(decision.snapshot)
        return min(max(1, decision.selectedChunkSize), requested)
    }
}
