import Foundation
import MLX
#if canImport(Darwin)
import Darwin
#endif

internal struct MLXRuntimeMemorySnapshot: Equatable, Sendable {
    let currentMemoryBytes: Int64
    let cacheMemoryBytes: Int64
    let physicalMemoryBytes: Int64
    let metalLimitBytes: Int64?
    let availableMemoryBytes: Int64?

    init(
        currentMemoryBytes: Int64,
        cacheMemoryBytes: Int64,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?,
        availableMemoryBytes: Int64? = nil
    ) {
        self.currentMemoryBytes = max(0, currentMemoryBytes)
        self.cacheMemoryBytes = max(0, cacheMemoryBytes)
        self.physicalMemoryBytes = max(0, physicalMemoryBytes)
        self.metalLimitBytes = metalLimitBytes.map { max(0, $0) }
        self.availableMemoryBytes = availableMemoryBytes.map { max(0, $0) }
    }

    static func live(
        metalLimitBytes: Int64? = MLXRuntimeMemoryGuard.metalRecommendedWorkingSetBytes()
    ) -> MLXRuntimeMemorySnapshot {
        let host = MLXHostMemorySnapshot.live()
        return MLXRuntimeMemorySnapshot(
            currentMemoryBytes: Int64(Memory.activeMemory),
            cacheMemoryBytes: Int64(Memory.cacheMemory),
            physicalMemoryBytes: host.physicalMemoryBytes,
            metalLimitBytes: metalLimitBytes,
            availableMemoryBytes: host.availableMemoryBytes
        )
    }

    func replacingCurrentMemoryBytes(_ bytes: Int64) -> MLXRuntimeMemorySnapshot {
        MLXRuntimeMemorySnapshot(
            currentMemoryBytes: bytes,
            cacheMemoryBytes: cacheMemoryBytes,
            physicalMemoryBytes: physicalMemoryBytes,
            metalLimitBytes: metalLimitBytes,
            availableMemoryBytes: availableMemoryBytes
        )
    }
}

private struct MLXHostMemorySnapshot: Equatable, Sendable {
    let physicalMemoryBytes: Int64
    let availableMemoryBytes: Int64?

    static func live() -> MLXHostMemorySnapshot {
        MLXHostMemorySnapshot(
            physicalMemoryBytes: Int64(ProcessInfo.processInfo.physicalMemory),
            availableMemoryBytes: availableBytes()
        )
    }

    private static func availableBytes() -> Int64? {
        #if canImport(Darwin)
        var pageSize = vm_size_t()
        let host = mach_host_self()
        guard host_page_size(host, &pageSize) == KERN_SUCCESS, pageSize > 0 else {
            return nil
        }

        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride
        )
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { rebound in
                host_statistics64(host, HOST_VM_INFO64, rebound, &count)
            }
        }
        guard result == KERN_SUCCESS else {
            return nil
        }

        let pages = UInt64(stats.free_count)
            + UInt64(stats.inactive_count)
            + UInt64(stats.speculative_count)
            + UInt64(stats.purgeable_count)
        let bytes = pages * UInt64(pageSize)
        return bytes >= UInt64(Int64.max) ? Int64.max : Int64(bytes)
        #else
        return nil
        #endif
    }
}

internal struct MLXModelMemoryProfile: Equatable, Sendable {
    private static let defaultDTypeSize = 2.0

    let numLayers: Int
    let numKVHeads: Int
    let numAttentionHeads: Int
    let headDimension: Int
    let dtypeSize: Double
    let scoreDTypeSize: Double
    let kvBytesPerTokenOverride: Double?
    var mlaBaseBytesPerToken: Double? = nil
    var mlaIndexHeadDimension: Int? = nil

    static func load(modelDirectory: URL) throws -> MLXModelMemoryProfile? {
        let configURL = modelDirectory.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            return nil
        }
        let data = try Data(contentsOf: configURL)
        guard let config = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return profile(from: config)
    }

    static func profile(from config: [String: Any]) -> MLXModelMemoryProfile? {
        let modelConfig = languageModelConfig(from: config)
        guard let numLayers = positiveInt(
            modelConfig,
            keys: ["num_hidden_layers", "n_layer", "num_layers"]
        ),
            let numAttentionHeads = positiveInt(
                modelConfig,
                keys: ["num_attention_heads", "n_head", "n_heads"]
            )
        else {
            return nil
        }
        let numKVHeads = positiveInt(
            modelConfig,
            keys: ["num_key_value_heads", "num_kv_heads"]
        ) ?? numAttentionHeads
        let headDimension = resolvedHeadDimension(
            config: modelConfig,
            attentionHeads: numAttentionHeads
        )
        guard let headDimension else {
            return nil
        }

        let dtypeSize = resolvedDTypeSize(config: modelConfig)
        let mlaLayout = mlaMemoryLayout(
            config: modelConfig,
            numLayers: numLayers,
            dtypeSize: dtypeSize
        )
        return MLXModelMemoryProfile(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            numAttentionHeads: numAttentionHeads,
            headDimension: headDimension,
            dtypeSize: dtypeSize,
            scoreDTypeSize: dtypeSize,
            kvBytesPerTokenOverride: mlaLayout?.bytesPerToken,
            mlaBaseBytesPerToken: mlaLayout?.baseBytesPerToken,
            mlaIndexHeadDimension: mlaLayout?.indexHeadDimension
        )
    }

    func estimatePromptKVBytes(tokenCount: Int) -> Int64 {
        guard tokenCount > 0 else {
            return 0
        }
        if let kvBytesPerTokenOverride {
            return Self.byteCount(Double(tokenCount) * kvBytesPerTokenOverride)
        }
        let bytes = Double(tokenCount)
            * Double(numLayers)
            * Double(numKVHeads)
            * Double(headDimension)
            * dtypeSize
            * 2.0
        return Self.byteCount(bytes)
    }

    func estimatePrefillPeakBytes(
        newTokenCount: Int,
        cachedTokenCount: Int,
        prefillStepSize: Int
    ) -> Int64 {
        guard newTokenCount > 0 else {
            return 0
        }
        let effectiveChunk = min(max(1, prefillStepSize), newTokenCount)
        let totalKVLength = newTokenCount + max(0, cachedTokenCount)
        return estimatePromptKVBytes(tokenCount: newTokenCount)
            + estimateSDPAActivationBytes(queryTokenCount: effectiveChunk, kvLength: totalKVLength)
    }

    func estimateGenerationPeakBytes(
        promptTokenCount: Int,
        cachedTokenCount: Int,
        maximumGeneratedTokenCount: Int,
        prefillStepSize: Int
    ) -> Int64 {
        let promptCount = max(promptTokenCount, 0)
        let cachedCount = min(max(cachedTokenCount, 0), promptCount)
        let newPromptCount = promptCount - cachedCount
        let generatedCount = max(maximumGeneratedTokenCount, 0)
        let prefillPeak = estimatePrefillPeakBytes(
            newTokenCount: newPromptCount,
            cachedTokenCount: cachedCount,
            prefillStepSize: prefillStepSize
        )
        guard generatedCount > 0 else {
            return prefillPeak
        }
        let promptKVBytes = estimatePromptKVBytes(tokenCount: newPromptCount)
        let generatedKVBytes = estimatePromptKVBytes(tokenCount: generatedCount)
        let decodeActivationBytes = estimateSDPAActivationBytes(
            queryTokenCount: 1,
            kvLength: promptCount + generatedCount
        )
        return max(prefillPeak, promptKVBytes + generatedKVBytes + decodeActivationBytes)
    }

    func estimateSDPAActivationBytes(queryTokenCount: Int, kvLength: Int) -> Int64 {
        guard queryTokenCount > 0, kvLength > 0 else {
            return 0
        }
        let output = Double(numAttentionHeads)
            * Double(queryTokenCount)
            * Double(headDimension)
            * 4.0
        guard !usesFusedSDPA(queryTokenCount: queryTokenCount, kvLength: kvLength) else {
            return Self.byteCount(output)
        }
        let scores = Double(numAttentionHeads)
            * Double(queryTokenCount)
            * Double(kvLength)
            * scoreDTypeSize
        return Self.byteCount(output + scores)
    }

    func applyingKVCacheBits(_ bits: Double?) -> MLXModelMemoryProfile {
        guard let bits, bits.isFinite, bits > 0 else {
            return self
        }
        let kvDTypeSize = bits / 8.0
        let scaledOverride = kvBytesPerTokenOverride.map {
            $0 * kvDTypeSize / max(Self.defaultDTypeSize, dtypeSize)
        }
        return MLXModelMemoryProfile(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            numAttentionHeads: numAttentionHeads,
            headDimension: headDimension,
            dtypeSize: kvDTypeSize,
            scoreDTypeSize: scoreDTypeSize,
            kvBytesPerTokenOverride: scaledOverride,
            mlaBaseBytesPerToken: mlaBaseBytesPerToken.map {
                $0 * kvDTypeSize / max(Self.defaultDTypeSize, dtypeSize)
            },
            mlaIndexHeadDimension: mlaIndexHeadDimension
        )
    }

    func refinedWithLiveCacheLayout(_ cache: [KVCache]) -> MLXModelMemoryProfile {
        guard let mlaBaseBytesPerToken,
            let mlaIndexHeadDimension,
            cache.count == numLayers
        else {
            return self
        }
        let liveIndexerLayerCount = Self.liveDSAIndexerLayerCount(cache)
        guard liveIndexerLayerCount > 0 else {
            return self
        }
        let liveIndexerBytes = Double(liveIndexerLayerCount * mlaIndexHeadDimension) * dtypeSize
        return MLXModelMemoryProfile(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            numAttentionHeads: numAttentionHeads,
            headDimension: headDimension,
            dtypeSize: dtypeSize,
            scoreDTypeSize: scoreDTypeSize,
            kvBytesPerTokenOverride: mlaBaseBytesPerToken + liveIndexerBytes,
            mlaBaseBytesPerToken: mlaBaseBytesPerToken,
            mlaIndexHeadDimension: mlaIndexHeadDimension
        )
    }

    private static func resolvedHeadDimension(
        config: [String: Any],
        attentionHeads: Int
    ) -> Int? {
        if let explicit = positiveInt(config, keys: ["head_dim", "head_dimensions"]) {
            return explicit
        }
        guard let hiddenSize = positiveInt(config, keys: ["hidden_size", "n_embd"]) else {
            return nil
        }
        return hiddenSize / attentionHeads
    }

    private func usesFusedSDPA(queryTokenCount: Int, kvLength: Int) -> Bool {
        // Current MLX fused SDPA tiles arbitrary head dimensions, so the
        // estimator should not model a materialized score matrix by head size.
        numAttentionHeads > 0
            && numKVHeads > 0
            && headDimension > 0
            && queryTokenCount > 0
            && kvLength >= queryTokenCount
    }

    private static func resolvedDTypeSize(config: [String: Any]) -> Double {
        let dtype = string(config, keys: ["torch_dtype", "dtype", "model_dtype"])?.lowercased() ?? ""
        if dtype.contains("float32") || dtype.contains("fp32") {
            return 4.0
        }
        return Self.defaultDTypeSize
    }

    private static func languageModelConfig(from config: [String: Any]) -> [String: Any] {
        for key in ["text_config", "language_config", "llm_config"] {
            guard let nested = config[key] as? [String: Any] else {
                continue
            }
            if positiveInt(nested, keys: ["num_hidden_layers", "n_layer", "num_layers"]) != nil {
                return nested
            }
        }
        return config
    }

    private static func mlaMemoryLayout(
        config: [String: Any],
        numLayers: Int,
        dtypeSize: Double
    ) -> (baseBytesPerToken: Double, indexHeadDimension: Int?, bytesPerToken: Double)? {
        guard let kvLoraRank = positiveInt(config, keys: ["kv_lora_rank"]),
            let ropeHeadDimension = positiveInt(config, keys: ["qk_rope_head_dim"])
        else {
            return nil
        }
        let mainCacheElements = numLayers * (kvLoraRank + ropeHeadDimension)
        let indexHeadDimension = positiveInt(config, keys: ["index_head_dim"])
        let dsaIndexerElements = dsaIndexerLayerCount(
            config: config,
            numLayers: numLayers
        ) * (indexHeadDimension ?? 0)
        let elementsPerToken = Double(mainCacheElements + dsaIndexerElements)
        return (
            baseBytesPerToken: Double(mainCacheElements) * dtypeSize,
            indexHeadDimension: indexHeadDimension,
            bytesPerToken: elementsPerToken * dtypeSize
        )
    }

    private static func liveDSAIndexerLayerCount(_ cache: [KVCache]) -> Int {
        cache.reduce(into: 0) { count, layerCache in
            guard let cacheList = layerCache as? CacheList,
                cacheList.layoutCaches.count > 1
            else {
                return
            }
            count += 1
        }
    }

    private static func dsaIndexerLayerCount(
        config: [String: Any],
        numLayers: Int
    ) -> Int {
        guard positiveInt(config, keys: ["index_head_dim"]) != nil else {
            return 0
        }
        if let types = stringArray(config["indexer_types"]), types.count == numLayers {
            return types.filter(isFullIndexer).count
        }
        if let pattern = config["index_topk_pattern"] {
            if let text = pattern as? String, text.count == numLayers {
                return text.map(String.init).filter(isFullIndexer).count
            }
            if let types = stringArray(pattern), types.count == numLayers {
                return types.filter(isFullIndexer).count
            }
        }
        guard positiveInt(config, keys: ["index_topk"]) != nil else {
            return 0
        }
        let frequency = max(positiveInt(config, keys: ["index_topk_freq"]) ?? 1, 1)
        let offset = positiveInt(config, keys: ["index_skip_topk_offset"]) ?? 2
        return (0 ..< numLayers).filter { layerIndex in
            max(layerIndex - offset + 1, 0) % frequency == 0
        }.count
    }

    private static func stringArray(_ value: Any?) -> [String]? {
        guard let values = value as? [Any] else {
            return value as? [String]
        }
        return values.compactMap { $0 as? String }
    }

    private static func isFullIndexer(_ value: String) -> Bool {
        let normalized = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return normalized == "full" || normalized == "f"
    }

    private static func positiveInt(_ config: [String: Any], keys: [String]) -> Int? {
        for key in keys {
            if let value = config[key] as? Int, value > 0 {
                return value
            }
            if let value = config[key] as? Double, value > 0 {
                return Int(value)
            }
            if let value = config[key] as? String, let parsed = Int(value), parsed > 0 {
                return parsed
            }
        }
        return nil
    }

    private static func string(_ config: [String: Any], keys: [String]) -> String? {
        for key in keys {
            if let value = config[key] as? String {
                return value
            }
        }
        return nil
    }

    private static func byteCount(_ value: Double) -> Int64 {
        guard value.isFinite, value > 0 else {
            return 0
        }
        return value >= Double(Int64.max) ? Int64.max : Int64(value.rounded(.up))
    }
}

internal enum MLXRuntimeMemoryGuard {
    private struct LimitDecision {
        let bytes: Int64
        let source: MLXMemoryGuardSnapshot.LimitSource
        let hardLimitFraction: Double
    }

    private static let oneGiB: Int64 = 1_073_741_824
    private static let modelArtifactExtensions: Set<String> = [
        "bin",
        "gguf",
        "mlx",
        "npz",
        "pt",
        "pth",
        "safetensors"
    ]
    private static let smallSystemThreshold = 24 * oneGiB
    private static let smallSystemReserve = 4 * oneGiB
    private static let largeSystemReserves: [MLXMemoryGuardTier: Int64] = [
        .safe: 8 * oneGiB,
        .balanced: 6 * oneGiB,
        .aggressive: 4 * oneGiB,
        .custom: 2 * oneGiB
    ]

    static func preflight(
        configuration: MLXMemoryGuardConfiguration,
        profile: MLXModelMemoryProfile?,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        maximumGeneratedTokenCount: Int = 0,
        prefillStepSize: Int,
        memorySnapshot: MLXRuntimeMemorySnapshot = .live()
    ) throws {
        guard configuration.tier != .off else {
            record(stage: .disabled, configuration: configuration)
            return
        }
        guard let profile else {
            record(stage: .profileUnavailable, configuration: configuration)
            return
        }
        let current = memorySnapshot.currentMemoryBytes
            + (configuration.includeCacheMemory ? memorySnapshot.cacheMemoryBytes : 0)
        guard let limit = limitDecision(configuration: configuration, snapshot: memorySnapshot, current: current) else {
            record(stage: .limitUnavailable, configuration: configuration)
            return
        }

        let cachedCount = min(max(cachedTokenCount, 0), max(promptTokenCount, 0))
        let newTokenCount = max(promptTokenCount - cachedCount, 0)
        let peak = profile.estimateGenerationPeakBytes(
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedCount,
            maximumGeneratedTokenCount: maximumGeneratedTokenCount,
            prefillStepSize: prefillStepSize
        )
        let estimated = current + peak
        let stage: MLXMemoryGuardSnapshot.Stage = estimated > limit.bytes ? .rejected : .allowed
        let message = rejectionMessage(
            estimatedBytes: estimated,
            currentBytes: current,
            peakBytes: peak,
            limit: limit
        )
        let snapshot = MLXMemoryGuardSnapshot(
            stage: stage,
            tier: configuration.tier,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedCount,
            newTokenCount: newTokenCount,
            maximumGeneratedTokenCount: max(maximumGeneratedTokenCount, 0),
            prefillStepSize: prefillStepSize,
            currentMemoryBytes: current,
            estimatedPeakBytes: peak,
            limitBytes: limit.bytes,
            limitSource: limit.source,
            message: stage == .rejected ? message : nil
        )
        MLXGenerationDiagnostics.recordMemoryGuard(snapshot)

        if stage == .rejected {
            throw LLMError.invalidConfiguration(message)
        }
    }

    static func preflight(
        configuration: MLXMemoryGuardConfiguration,
        profile: MLXModelMemoryProfile?,
        promptTokenCount: Int,
        cachedTokenCount: Int,
        maximumGeneratedTokenCount: Int = 0,
        prefillStepSize: Int,
        currentMemoryBytes: Int64,
        cacheMemoryBytes: Int64,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?
    ) throws {
        try preflight(
            configuration: configuration,
            profile: profile,
            promptTokenCount: promptTokenCount,
            cachedTokenCount: cachedTokenCount,
            maximumGeneratedTokenCount: maximumGeneratedTokenCount,
            prefillStepSize: prefillStepSize,
            memorySnapshot: MLXRuntimeMemorySnapshot(
                currentMemoryBytes: currentMemoryBytes,
                cacheMemoryBytes: cacheMemoryBytes,
                physicalMemoryBytes: physicalMemoryBytes,
                metalLimitBytes: metalLimitBytes
            )
        )
    }

    static func preflightModelLoad(
        configuration: MLXMemoryGuardConfiguration,
        modelDirectory: URL,
        memorySnapshot: MLXRuntimeMemorySnapshot = .live()
    ) throws {
        let modelLoadBytes = try estimatedModelLoadBytes(modelDirectory: modelDirectory)
        try preflightModelLoad(
            configuration: configuration,
            modelLoadBytes: modelLoadBytes,
            memorySnapshot: memorySnapshot
        )
    }

    static func preflightModelLoad(
        configuration: MLXMemoryGuardConfiguration,
        modelDirectory: URL,
        currentMemoryBytes: Int64,
        cacheMemoryBytes: Int64,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?
    ) throws {
        try preflightModelLoad(
            configuration: configuration,
            modelDirectory: modelDirectory,
            memorySnapshot: MLXRuntimeMemorySnapshot(
                currentMemoryBytes: currentMemoryBytes,
                cacheMemoryBytes: cacheMemoryBytes,
                physicalMemoryBytes: physicalMemoryBytes,
                metalLimitBytes: metalLimitBytes
            )
        )
    }

    static func preflightModelLoad(
        configuration: MLXMemoryGuardConfiguration,
        modelLoadBytes: Int64?,
        memorySnapshot: MLXRuntimeMemorySnapshot = .live()
    ) throws {
        guard configuration.tier != .off else {
            record(stage: .disabled, configuration: configuration)
            return
        }
        guard let modelLoadBytes, modelLoadBytes > 0 else {
            record(stage: .modelLoadEstimateUnavailable, configuration: configuration)
            return
        }
        let current = memorySnapshot.currentMemoryBytes
            + (configuration.includeCacheMemory ? memorySnapshot.cacheMemoryBytes : 0)
        guard let limit = limitDecision(configuration: configuration, snapshot: memorySnapshot, current: current) else {
            record(stage: .limitUnavailable, configuration: configuration)
            return
        }

        let estimated = current + modelLoadBytes
        let stage: MLXMemoryGuardSnapshot.Stage =
            estimated > limit.bytes ? .modelLoadRejected : .modelLoadAllowed
        let message = modelLoadRejectionMessage(
            estimatedBytes: estimated,
            currentBytes: current,
            modelLoadBytes: modelLoadBytes,
            limit: limit
        )
        MLXGenerationDiagnostics.recordMemoryGuard(MLXMemoryGuardSnapshot(
            stage: stage,
            tier: configuration.tier,
            promptTokenCount: 0,
            cachedTokenCount: 0,
            newTokenCount: 0,
            maximumGeneratedTokenCount: 0,
            prefillStepSize: 0,
            currentMemoryBytes: current,
            estimatedPeakBytes: modelLoadBytes,
            limitBytes: limit.bytes,
            limitSource: limit.source,
            message: stage == .modelLoadRejected ? message : nil
        ))

        if stage == .modelLoadRejected {
            throw LLMError.invalidConfiguration(message)
        }
    }

    static func preflightModelLoad(
        configuration: MLXMemoryGuardConfiguration,
        modelLoadBytes: Int64?,
        currentMemoryBytes: Int64,
        cacheMemoryBytes: Int64,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?
    ) throws {
        try preflightModelLoad(
            configuration: configuration,
            modelLoadBytes: modelLoadBytes,
            memorySnapshot: MLXRuntimeMemorySnapshot(
                currentMemoryBytes: currentMemoryBytes,
                cacheMemoryBytes: cacheMemoryBytes,
                physicalMemoryBytes: physicalMemoryBytes,
                metalLimitBytes: metalLimitBytes
            )
        )
    }

    static func estimatedModelLoadBytes(modelDirectory: URL) throws -> Int64? {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(
            atPath: modelDirectory.path,
            isDirectory: &isDirectory
        ) else {
            return nil
        }

        if !isDirectory.boolValue {
            return try modelArtifactByteCount(for: modelDirectory)
        }

        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: [.fileSizeKey, .isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return nil
        }

        var byteCount: Int64 = 0
        for case let url as URL in enumerator {
            guard let fileBytes = try modelArtifactByteCount(for: url) else {
                continue
            }
            byteCount += fileBytes
        }
        return byteCount > 0 ? byteCount : nil
    }

    static func limitBytes(
        configuration: MLXMemoryGuardConfiguration,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?
    ) -> Int64? {
        limitDecision(
            configuration: configuration,
            physicalMemoryBytes: physicalMemoryBytes,
            metalLimitBytes: metalLimitBytes
        )?.bytes
    }

    static func limitBytes(
        configuration: MLXMemoryGuardConfiguration,
        memorySnapshot: MLXRuntimeMemorySnapshot,
        currentMemoryBytes: Int64
    ) -> Int64? {
        limitDecision(
            configuration: configuration,
            snapshot: memorySnapshot,
            current: currentMemoryBytes
        )?.bytes
    }

    private static func limitDecision(
        configuration: MLXMemoryGuardConfiguration,
        snapshot: MLXRuntimeMemorySnapshot,
        current: Int64
    ) -> LimitDecision? {
        limitDecision(
            configuration: configuration,
            currentMemoryBytes: current,
            physicalMemoryBytes: snapshot.physicalMemoryBytes,
            metalLimitBytes: snapshot.metalLimitBytes,
            availableMemoryBytes: snapshot.availableMemoryBytes
        )
    }

    private static func limitDecision(
        configuration: MLXMemoryGuardConfiguration,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?
    ) -> LimitDecision? {
        limitDecision(
            configuration: configuration,
            currentMemoryBytes: nil,
            physicalMemoryBytes: physicalMemoryBytes,
            metalLimitBytes: metalLimitBytes,
            availableMemoryBytes: nil
        )
    }

    private static func limitDecision(
        configuration: MLXMemoryGuardConfiguration,
        currentMemoryBytes: Int64?,
        physicalMemoryBytes: Int64,
        metalLimitBytes: Int64?,
        availableMemoryBytes: Int64?
    ) -> LimitDecision? {
        guard configuration.tier != .off, physicalMemoryBytes > 0 else {
            return nil
        }
        let reserve = reserveBytes(
            tier: configuration.tier,
            physicalMemoryBytes: physicalMemoryBytes
        )
        var limit = max(1, physicalMemoryBytes - reserve)
        var source = MLXMemoryGuardSnapshot.LimitSource.processMemoryBudget
        if configuration.tier == .custom, let customLimit = configuration.customLimitBytes, customLimit > 0 {
            if customLimit < limit {
                limit = customLimit
                source = .customLimit
            }
        }
        if let metalLimitBytes, metalLimitBytes > 0 {
            if metalLimitBytes < limit {
                limit = metalLimitBytes
                source = .metalRecommendedWorkingSet
            }
        }
        if let currentMemoryBytes, let availableMemoryBytes, availableMemoryBytes > 0 {
            let hostLimit = max(1, max(0, currentMemoryBytes) + availableMemoryBytes)
            if hostLimit < limit {
                limit = hostLimit
                source = .hostAvailableMemory
            }
        }
        let fraction = min(1.0, max(0.1, configuration.hardLimitFraction))
        return LimitDecision(
            bytes: max(1, Int64((Double(limit) * fraction).rounded(.down))),
            source: source,
            hardLimitFraction: fraction
        )
    }

    private static func reserveBytes(
        tier: MLXMemoryGuardTier,
        physicalMemoryBytes: Int64
    ) -> Int64 {
        if physicalMemoryBytes < smallSystemThreshold {
            return smallSystemReserve
        }
        return largeSystemReserves[tier] ?? largeSystemReserves[.balanced]!
    }

    private static func record(
        stage: MLXMemoryGuardSnapshot.Stage,
        configuration: MLXMemoryGuardConfiguration
    ) {
        MLXGenerationDiagnostics.recordMemoryGuard(MLXMemoryGuardSnapshot(
            stage: stage,
            tier: configuration.tier,
            promptTokenCount: 0,
            cachedTokenCount: 0,
            newTokenCount: 0,
            maximumGeneratedTokenCount: 0,
            prefillStepSize: 0,
            currentMemoryBytes: nil,
            estimatedPeakBytes: nil,
            limitBytes: nil,
            limitSource: nil,
            message: nil
        ))
    }

    private static func rejectionMessage(
        estimatedBytes: Int64,
        currentBytes: Int64,
        peakBytes: Int64,
        limit: LimitDecision
    ) -> String {
        """
        Generation would require about \(format(estimatedBytes)) peak \
        (current \(format(currentBytes)) + prompt/decode KV and SDPA \(format(peakBytes))), \
        but the active \(limitDescription(limit)) is \(format(limit.bytes)). \
        Reduce context length, unload another model, or use a less conservative memory guard tier.
        """
    }

    private static func limitDescription(_ limit: LimitDecision) -> String {
        let base = switch limit.source {
        case .processMemoryBudget:
            "process memory budget"

        case .customLimit:
            "custom memory guard limit"

        case .metalRecommendedWorkingSet:
            "Metal recommended working set ceiling"

        case .hostAvailableMemory:
            "host available memory ceiling"
        }
        guard limit.hardLimitFraction < 1 else {
            return base
        }
        return "\(base) after \(Int((limit.hardLimitFraction * 100).rounded()))% hard-limit scaling"
    }

    private static func modelArtifactByteCount(for url: URL) throws -> Int64? {
        guard modelArtifactExtensions.contains(url.pathExtension.lowercased()) else {
            return nil
        }
        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        guard values.isRegularFile == true, let fileSize = values.fileSize, fileSize > 0 else {
            return nil
        }
        return Int64(fileSize)
    }

    private static func modelLoadRejectionMessage(
        estimatedBytes: Int64,
        currentBytes: Int64,
        modelLoadBytes: Int64,
        limit: LimitDecision
    ) -> String {
        """
        Model load would require about \(format(estimatedBytes)) peak \
        (current \(format(currentBytes)) + model weights \(format(modelLoadBytes))), \
        but the active \(limitDescription(limit)) is \(format(limit.bytes)). \
        Unload another model, choose a smaller quantization, or use a less conservative memory guard tier.
        """
    }

    private static func format(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .memory)
    }

    static func metalRecommendedWorkingSetBytes() -> Int64? {
        guard let bytes = GPU.maxRecommendedWorkingSetBytes() else {
            return nil
        }
        return Int64(bytes)
    }
}
