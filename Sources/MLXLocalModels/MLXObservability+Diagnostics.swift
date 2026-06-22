import Foundation

internal extension MLXObservability {
    static func recordDiagnosticEvent(_ event: MLXGenerationDiagnosticEvent) {
        switch event {
        case .parameters(let snapshot):
            recordParameters(snapshot)
        case .promptCachePlan(let snapshot):
            recordPromptCachePlan(snapshot)
        case .speculativeDecoding(let snapshot):
            setGauge(
                "generation.speculative.draft_tokens",
                value: Double(snapshot.numDraftTokens),
                category: .generation
            )
        case .specPrefillPlan(let snapshot):
            recordSpecPrefillPlan(snapshot)
        case .dFlashPlan(let snapshot):
            log(
                "generation.dflash.\(snapshot.stage.rawValue)",
                category: .generation,
                severity: .debug,
                attributes: [
                    "stage": snapshot.stage.rawValue,
                    "verify_mode": snapshot.verifyMode.map(String.init(describing:)) ?? "none"
                ],
                measurements: compactMeasurements([
                    "prompt_tokens": Double(snapshot.promptTokenCount),
                    "draft_window_size": snapshot.draftWindowSize.map(Double.init),
                    "draft_sink_size": snapshot.draftSinkSize.map(Double.init)
                ])
            )
        case .adaptivePrefillChunk(let snapshot):
            recordAdaptivePrefillChunk(snapshot)
        case .prefillChunk(let snapshot):
            recordPrefillChunk(snapshot)
        case .cacheSnapshot(let snapshot):
            recordCacheSnapshot(snapshot)
        case .quantizedKVConversion(let snapshot):
            incrementCounter(
                "generation.kv_cache.quantized_conversions",
                by: Double(snapshot.convertedCount),
                category: .generation,
                attributes: ["bits": String(snapshot.kvBits)]
            )
        case .grammarConstraint(let snapshot):
            recordGrammarConstraint(snapshot)
        case .reasoningBudget(let snapshot):
            recordReasoningBudget(snapshot)
        case .generatedToken:
            break
        case .memoryGuard(let snapshot):
            recordMemoryGuard(snapshot)
        case .executionPlan(let snapshot):
            recordExecutionPlan(snapshot)
        case .admission(let snapshot):
            recordAdmission(snapshot)
        case .batchRows(let snapshot):
            setGauge(
                "generation.continuous_batch.rows",
                value: Double(snapshot.rowCount),
                category: .generation,
                attributes: ["stage": snapshot.stage.rawValue]
            )
        case .continuousBatchLogits(let snapshot):
            incrementCounter(
                "generation.continuous_batch.logit_rows",
                by: Double(snapshot.rowCount),
                category: .generation,
                attributes: ["stage": snapshot.stage.rawValue]
            )
        case .pagedKVBlocks(let snapshot):
            recordPagedKVBlocks(snapshot)
        case .persistentCacheInvalidation(let snapshot):
            incrementCounter(
                "prompt_cache.persistent.removed",
                by: Double(snapshot.removedCount),
                category: .promptCache,
                attributes: ["stage": snapshot.stage.rawValue]
            )
        case .promptCacheLookup(let snapshot):
            recordPromptCacheLookup(snapshot)
        case .promptCacheObservability(let snapshot):
            recordPromptCacheObservability(snapshot)
        case .sessionLifecycle(let snapshot):
            recordSessionLifecycle(snapshot)
        }
    }

    private static func recordParameters(_ snapshot: MLXGenerationParameterSnapshot) {
        log(
            "generation.parameters",
            category: .generation,
            severity: .debug,
            attributes: [
                "grammar": snapshot.grammarKind.map(\.rawValue) ?? "none",
                "mirostat": snapshot.mirostatVersion.map(String.init(describing:)) ?? "off"
            ],
            measurements: compactMeasurements([
                "max_tokens": snapshot.maxTokens.map(Double.init),
                "temperature": Double(snapshot.temperature),
                "top_p": Double(snapshot.topP),
                "top_k": Double(snapshot.topK),
                "prefill_step_size": Double(snapshot.prefillStepSize),
                "kv_bits": snapshot.kvBits.map(Double.init)
            ])
        )
    }

    private static func recordPromptCachePlan(_ snapshot: MLXPromptCachePlanSnapshot) {
        incrementCounter("prompt_cache.prefix.requests", category: .promptCache)
        if snapshot.reusedTokenCount > 0 {
            incrementCounter("prompt_cache.prefix.hits", category: .promptCache)
        } else {
            incrementCounter("prompt_cache.prefix.misses", category: .promptCache)
        }
        incrementCounter(
            "prompt_cache.tokens.requested",
            by: Double(max(snapshot.promptTokenCount - 1, 0)),
            category: .promptCache
        )
        incrementCounter(
            "prompt_cache.tokens.reused",
            by: Double(snapshot.reusedTokenCount),
            category: .promptCache
        )
        recordHistogram(
            "prompt_cache.prefix.reused_tokens",
            value: Double(snapshot.reusedTokenCount),
            category: .promptCache
        )
    }

    private static func recordPromptCacheLookup(_ snapshot: MLXPromptCacheLookupSnapshot) {
        incrementCounter(
            "prompt_cache.lookup.requests",
            category: .promptCache,
            attributes: ["strategy": snapshot.strategy.rawValue]
        )
        if snapshot.reusedTokenCount > 0 {
            incrementCounter(
                "prompt_cache.lookup.hits",
                category: .promptCache,
                attributes: ["strategy": snapshot.strategy.rawValue]
            )
        }
        recordHistogram(
            "prompt_cache.lookup.candidates",
            value: Double(snapshot.candidateCount),
            category: .promptCache,
            attributes: ["strategy": snapshot.strategy.rawValue]
        )
        recordHistogram(
            "prompt_cache.lookup.reused_tokens",
            value: Double(snapshot.reusedTokenCount),
            category: .promptCache,
            attributes: ["strategy": snapshot.strategy.rawValue]
        )
    }

    private static func recordPromptCacheObservability(
        _ snapshot: MLXPromptCacheObservabilitySnapshot
    ) {
        let counters = snapshot.counters
        setGauge("prompt_cache.prefix.hit_rate", value: counters.prefixHitRate, category: .promptCache)
        setGauge(
            "prompt_cache.prefix.match_efficiency",
            value: counters.prefixMatchEfficiency,
            category: .promptCache
        )
        setGauge("prompt_cache.ssd.hot_rate", value: counters.ssdHotRate, category: .promptCache)
        setGauge("prompt_cache.evictions.total", value: Double(counters.evictions), category: .promptCache)
        setGauge("prompt_cache.ssd.saves.total", value: Double(counters.ssdSaves), category: .promptCache)
        setGauge(
            "prompt_cache.hot.promotions.total",
            value: Double(counters.hotCachePromotions),
            category: .promptCache
        )
        for window in snapshot.windows.values {
            let attributes = ["window": window.label]
            setGauge(
                "prompt_cache.window.prefix_hit_rate",
                value: window.prefixHitRate,
                category: .promptCache,
                attributes: attributes
            )
            setGauge(
                "prompt_cache.window.eviction_rate_per_minute",
                value: window.evictionRatePerMinute,
                category: .promptCache,
                attributes: attributes
            )
            setGauge(
                "prompt_cache.window.ssd_hot_rate",
                value: window.ssdHotRate,
                category: .promptCache,
                attributes: attributes
            )
        }
    }

    private static func recordSpecPrefillPlan(_ snapshot: MLXSpecPrefillPlanSnapshot) {
        log(
            "generation.spec_prefill.\(snapshot.stage.rawValue)",
            category: .generation,
            severity: .debug,
            attributes: ["stage": snapshot.stage.rawValue],
            measurements: compactMeasurements([
                "prompt_tokens": Double(snapshot.promptTokenCount),
                "cached_tokens": Double(snapshot.cachedTokenCount),
                "new_prefill_tokens": Double(snapshot.newPrefillTokenCount),
                "retained_tokens": Double(snapshot.retainedTokenCount),
                "keep_rate": snapshot.keepRate
            ])
        )
    }

    private static func recordAdaptivePrefillChunk(_ snapshot: MLXAdaptivePrefillChunkSnapshot) {
        recordHistogram(
            "generation.prefill.selected_chunk_tokens",
            value: Double(snapshot.selectedChunkSize),
            category: .generation,
            attributes: ["stage": snapshot.stage.rawValue]
        )
        recordHistogram(
            "generation.prefill.requested_chunk_tokens",
            value: Double(snapshot.requestedChunkSize),
            category: .generation,
            attributes: ["stage": snapshot.stage.rawValue]
        )
        if let predictedTransientBytes = snapshot.predictedTransientBytes {
            setGauge(
                "generation.prefill.predicted_transient_bytes",
                value: Double(predictedTransientBytes),
                category: .generation,
                attributes: ["stage": snapshot.stage.rawValue]
            )
        }
    }

    private static func recordPrefillChunk(_ snapshot: MLXPrefillChunkSnapshot) {
        recordHistogram(
            "generation.prefill.chunk_tokens",
            value: Double(snapshot.chunkSize),
            category: .generation
        )
        if let memoryDeltaBytes = snapshot.memoryDeltaBytes {
            recordHistogram(
                "generation.prefill.memory_delta_bytes",
                value: Double(memoryDeltaBytes),
                category: .generation
            )
        }
    }

    private static func recordCacheSnapshot(_ snapshot: MLXCacheSnapshot) {
        setGauge(
            "generation.kv_cache.entries",
            value: Double(snapshot.entries.count),
            category: .generation,
            attributes: ["label": snapshot.label]
        )
        let maxOffset = snapshot.entries.map(\.offset).max() ?? 0
        setGauge(
            "generation.kv_cache.max_offset",
            value: Double(maxOffset),
            category: .generation,
            attributes: ["label": snapshot.label]
        )
    }

    private static func recordGrammarConstraint(_ snapshot: MLXGrammarConstraintSnapshot) {
        let attributes = [
            "stage": snapshot.stage.rawValue,
            "kind": snapshot.kind.map(\.rawValue) ?? "none"
        ]
        switch snapshot.stage {
        case .tokenAccepted:
            incrementCounter("grammar.tokens.accepted", category: .grammar, attributes: attributes)
        case .tokenRejected:
            incrementCounter("grammar.tokens.rejected", category: .grammar, attributes: attributes)
        case .maskApplied, .batchMaskApplied, .mlxMaskPrepared, .mlxMaskReused:
            incrementCounter("grammar.masks.applied", category: .grammar, attributes: attributes)
        case .processorFailedClosed:
            incrementCounter("grammar.fail_closed", category: .grammar, attributes: attributes)
        default:
            log(
                "grammar.\(snapshot.stage.rawValue)",
                category: .grammar,
                severity: .debug,
                attributes: attributes,
                measurements: compactMeasurements([
                    "token_count": snapshot.tokenCount.map(Double.init),
                    "vocabulary_size": snapshot.vocabularySize.map(Double.init),
                    "bitmask_size": snapshot.bitmaskSize.map(Double.init)
                ])
            )
        }
    }

    private static func recordReasoningBudget(_ snapshot: MLXReasoningBudgetSnapshot) {
        incrementCounter(
            "generation.reasoning.\(snapshot.stage.rawValue)",
            category: .generation
        )
        setGauge(
            "generation.reasoning.tokens",
            value: Double(snapshot.reasoningTokenCount),
            category: .generation,
            attributes: ["stage": snapshot.stage.rawValue]
        )
    }

    private static func recordMemoryGuard(_ snapshot: MLXMemoryGuardSnapshot) {
        let attributes = [
            "stage": snapshot.stage.rawValue,
            "tier": snapshot.tier.rawValue,
            "limit_source": snapshot.limitSource?.rawValue ?? "none"
        ]
        if let currentMemoryBytes = snapshot.currentMemoryBytes {
            setGauge(
                "memory.current_bytes",
                value: Double(currentMemoryBytes),
                category: .memoryGuard,
                attributes: attributes
            )
        }
        if let estimatedPeakBytes = snapshot.estimatedPeakBytes {
            setGauge(
                "memory.estimated_peak_bytes",
                value: Double(estimatedPeakBytes),
                category: .memoryGuard,
                attributes: attributes
            )
        }
        if let limitBytes = snapshot.limitBytes {
            setGauge(
                "memory.limit_bytes",
                value: Double(limitBytes),
                category: .memoryGuard,
                attributes: attributes
            )
        }
        if snapshot.stage == .rejected || snapshot.stage == .modelLoadRejected {
            incrementCounter("memory_guard.rejections", category: .memoryGuard, attributes: attributes)
            log(
                "memory_guard.rejected",
                category: .memoryGuard,
                severity: .warning,
                attributes: attributes,
                measurements: compactMeasurements([
                    "prompt_tokens": Double(snapshot.promptTokenCount),
                    "cached_tokens": Double(snapshot.cachedTokenCount),
                    "new_tokens": Double(snapshot.newTokenCount),
                    "max_generated_tokens": Double(snapshot.maximumGeneratedTokenCount),
                    "estimated_peak_bytes": snapshot.estimatedPeakBytes.map(Double.init),
                    "limit_bytes": snapshot.limitBytes.map(Double.init)
                ])
            )
        }
    }

    private static func recordExecutionPlan(_ snapshot: MLXGenerationExecutionPlanSnapshot) {
        log(
            "generation.execution_plan",
            category: .generation,
            severity: .info,
            attributes: [
                "requested_strategy": String(describing: snapshot.requestedStrategy),
                "selected_strategy": String(describing: snapshot.selectedStrategy),
                "reason": String(describing: snapshot.reason),
                "supports_continuous_batching": String(snapshot.supportsContinuousBatching)
            ],
            measurements: [
                "max_concurrent_requests": Double(snapshot.effectiveMaxConcurrentRequests),
                "max_batch_size": Double(snapshot.effectiveMaxBatchSize)
            ]
        )
    }

    private static func recordAdmission(_ snapshot: MLXGenerationAdmissionSnapshot) {
        let attributes = [
            "stage": snapshot.stage.rawValue,
            "paused": String(snapshot.admissionPaused)
        ]
        setGauge(
            "admission.active",
            value: Double(snapshot.activeCount),
            category: .admission,
            attributes: attributes
        )
        setGauge(
            "admission.waiting",
            value: Double(snapshot.waitingCount),
            category: .admission,
            attributes: attributes
        )
        if snapshot.stage == .queueFull {
            incrementCounter("admission.queue_full", category: .admission, attributes: attributes)
        }
    }

    private static func recordPagedKVBlocks(_ snapshot: MLXPagedKVBlockTableSnapshot) {
        let attributes = ["stage": snapshot.stage.rawValue]
        setGauge("kv_blocks.capacity", value: Double(snapshot.capacity), category: .generation, attributes: attributes)
        setGauge("kv_blocks.used", value: Double(snapshot.usedCount), category: .generation, attributes: attributes)
        setGauge("kv_blocks.free", value: Double(snapshot.freeCount), category: .generation, attributes: attributes)
        setGauge(
            "kv_blocks.evictable",
            value: Double(snapshot.evictableCount),
            category: .generation,
            attributes: attributes
        )
    }

    private static func recordSessionLifecycle(_ snapshot: MLXSessionLifecycleSnapshot) {
        let attributes = [
            "stage": snapshot.stage.rawValue,
            "has_model": String(snapshot.hasModelContainer),
            "pending_unload": String(snapshot.pendingUnloadAfterGeneration)
        ]
        setGauge(
            "session.active_generations",
            value: Double(snapshot.activeGenerationCount),
            category: .runtime,
            attributes: attributes
        )
        if snapshot.stage == .unloadDeferred {
            incrementCounter("session.unload_deferred", category: .runtime, attributes: attributes)
        }
    }

    private static func compactMeasurements(
        _ values: [String: Double?]
    ) -> [String: Double] {
        var result: [String: Double] = [:]
        for (key, value) in values {
            guard let value else {
                continue
            }
            result[key] = value
        }
        return result
    }
}
