import MLX

/// Shared behavior for text-only language models.
internal protocol LLMModel: LanguageModel, LoRAModel {}

extension LLMModel {
    /// Prefills the KV cache until the remaining prompt can be handed to the token iterator.
    internal func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult {
        let targetChunkSize = windowSize ?? GenerationConstants.defaultPrefillStepSize
        let controller = MLXGenerationDiagnostics.currentAdaptivePrefillController
        var pending = input.text

        while true {
            let remainingTokenCount = pending.tokens.size
            let chunkSize = Self.prefillChunkSize(
                remainingTokenCount: remainingTokenCount,
                targetChunkSize: targetChunkSize,
                controller: controller
            )

            guard remainingTokenCount > chunkSize else {
                return .tokens(pending)
            }

            let memoryBeforeBytes = Int64(Memory.activeMemory)
            let chunk = pending[.newAxis, ..<chunkSize]

            _ = self(chunk, cache: cache.isEmpty ? nil : cache, state: nil)
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
                prefillStepSize: targetChunkSize,
                memoryBeforeBytes: memoryBeforeBytes,
                memoryAfterBytes: memoryAfterBytes
            )
            pending = pending[chunkSize...]
        }
    }

    private static func prefillChunkSize(
        remainingTokenCount: Int,
        targetChunkSize: Int,
        controller: MLXAdaptivePrefillChunkController?
    ) -> Int {
        let requested = min(max(1, targetChunkSize), max(1, remainingTokenCount))

        guard let controller else {
            return requested
        }

        let decision = controller.decision(remainingTokenCount: remainingTokenCount)
        MLXGenerationDiagnostics.recordAdaptivePrefillChunk(decision.snapshot)
        return min(max(1, decision.selectedChunkSize), requested)
    }
}
