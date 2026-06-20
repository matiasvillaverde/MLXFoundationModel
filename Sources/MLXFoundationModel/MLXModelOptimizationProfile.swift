import Foundation
import MLXLocalModels

/// Optimization metadata inferred without loading model weights.
public struct MLXModelOptimizationProfile: Codable, Equatable, Hashable, Sendable {
    private enum CodingKeys: CodingKey {
        case quantization
        case isOQQuantized
        case oQLevel
        case requiresFP8ScaleDequantization
        case hasNativeMTPWeights
        case supportsNativeMTP
        case nativeMTPRuntimeSupported
        case supportsVLMMTP
        case supportsVLMMTPDrafter
        case supportsSpeculativePrefill
        case supportsDFlash
        case supportsIndexCache
        case supportsTurboQuantKV
        case promptCacheReuseAlignment
        case defaultIndexCacheFrequency
    }

    public let quantization: MLXModelQuantizationProfile?
    public let isOQQuantized: Bool
    public let oQLevel: String?
    public let requiresFP8ScaleDequantization: Bool
    public let hasNativeMTPWeights: Bool
    public let supportsNativeMTP: Bool
    public let nativeMTPRuntimeSupported: Bool
    public let supportsVLMMTP: Bool
    public let supportsVLMMTPDrafter: Bool
    public let supportsSpeculativePrefill: Bool
    public let supportsDFlash: Bool
    public let supportsIndexCache: Bool
    public let supportsTurboQuantKV: Bool
    public let promptCacheReuseAlignment: PromptCacheReuseAlignment?
    public let defaultIndexCacheFrequency: Int?

    public init(
        quantization: MLXModelQuantizationProfile? = nil,
        isOQQuantized: Bool = false,
        oQLevel: String? = nil,
        requiresFP8ScaleDequantization: Bool = false,
        hasNativeMTPWeights: Bool = false,
        supportsNativeMTP: Bool = false,
        nativeMTPRuntimeSupported: Bool = false,
        supportsVLMMTP: Bool = false,
        supportsVLMMTPDrafter: Bool = false,
        supportsSpeculativePrefill: Bool = false,
        supportsDFlash: Bool = false,
        supportsIndexCache: Bool = false,
        supportsTurboQuantKV: Bool = false,
        promptCacheReuseAlignment: PromptCacheReuseAlignment? = nil,
        defaultIndexCacheFrequency: Int? = nil
    ) {
        self.quantization = quantization
        self.isOQQuantized = isOQQuantized
        self.oQLevel = oQLevel
        self.requiresFP8ScaleDequantization = requiresFP8ScaleDequantization
        self.hasNativeMTPWeights = hasNativeMTPWeights
        self.supportsNativeMTP = supportsNativeMTP
        self.nativeMTPRuntimeSupported = nativeMTPRuntimeSupported
        self.supportsVLMMTP = supportsVLMMTP
        self.supportsVLMMTPDrafter = supportsVLMMTPDrafter
        self.supportsSpeculativePrefill = supportsSpeculativePrefill
        self.supportsDFlash = supportsDFlash
        self.supportsIndexCache = supportsIndexCache
        self.supportsTurboQuantKV = supportsTurboQuantKV
        self.promptCacheReuseAlignment = promptCacheReuseAlignment
        self.defaultIndexCacheFrequency = defaultIndexCacheFrequency
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            quantization: try container.decodeIfPresent(
                MLXModelQuantizationProfile.self,
                forKey: .quantization
            ),
            isOQQuantized: try Self.decodeBool(container, .isOQQuantized),
            oQLevel: try container.decodeIfPresent(String.self, forKey: .oQLevel),
            requiresFP8ScaleDequantization: try Self.decodeBool(
                container,
                .requiresFP8ScaleDequantization
            ),
            hasNativeMTPWeights: try Self.decodeBool(container, .hasNativeMTPWeights),
            supportsNativeMTP: try Self.decodeBool(container, .supportsNativeMTP),
            nativeMTPRuntimeSupported: try Self.decodeBool(container, .nativeMTPRuntimeSupported),
            supportsVLMMTP: try Self.decodeBool(container, .supportsVLMMTP),
            supportsVLMMTPDrafter: try Self.decodeBool(container, .supportsVLMMTPDrafter),
            supportsSpeculativePrefill: try Self.decodeBool(container, .supportsSpeculativePrefill),
            supportsDFlash: try Self.decodeBool(container, .supportsDFlash),
            supportsIndexCache: try Self.decodeBool(container, .supportsIndexCache),
            supportsTurboQuantKV: try Self.decodeBool(container, .supportsTurboQuantKV),
            promptCacheReuseAlignment: try container.decodeIfPresent(
                PromptCacheReuseAlignment.self,
                forKey: .promptCacheReuseAlignment
            ),
            defaultIndexCacheFrequency: try container.decodeIfPresent(
                Int.self,
                forKey: .defaultIndexCacheFrequency
            )
        )
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(quantization, forKey: .quantization)
        try container.encode(isOQQuantized, forKey: .isOQQuantized)
        try container.encodeIfPresent(oQLevel, forKey: .oQLevel)
        try container.encode(requiresFP8ScaleDequantization, forKey: .requiresFP8ScaleDequantization)
        try container.encode(hasNativeMTPWeights, forKey: .hasNativeMTPWeights)
        try container.encode(supportsNativeMTP, forKey: .supportsNativeMTP)
        try container.encode(nativeMTPRuntimeSupported, forKey: .nativeMTPRuntimeSupported)
        try container.encode(supportsVLMMTP, forKey: .supportsVLMMTP)
        try container.encode(supportsVLMMTPDrafter, forKey: .supportsVLMMTPDrafter)
        try container.encode(supportsSpeculativePrefill, forKey: .supportsSpeculativePrefill)
        try container.encode(supportsDFlash, forKey: .supportsDFlash)
        try container.encode(supportsIndexCache, forKey: .supportsIndexCache)
        try container.encode(supportsTurboQuantKV, forKey: .supportsTurboQuantKV)
        try container.encodeIfPresent(promptCacheReuseAlignment, forKey: .promptCacheReuseAlignment)
        try container.encodeIfPresent(defaultIndexCacheFrequency, forKey: .defaultIndexCacheFrequency)
    }

    private static func decodeBool(
        _ container: KeyedDecodingContainer<CodingKeys>,
        _ key: CodingKeys
    ) throws -> Bool {
        try container.decodeIfPresent(Bool.self, forKey: key) ?? false
    }

    public static let empty = Self()

    public var detectedFeatures: Set<MLXModelOptimizationFeature> {
        featureSet(
            includingCandidateFeatures: true
        )
    }

    public var implementedFeatures: Set<MLXModelOptimizationFeature> {
        featureSet(
            includingCandidateFeatures: false
        )
    }

    public var pendingRuntimeFeatures: Set<MLXModelOptimizationFeature> {
        detectedFeatures.subtracting(implementedFeatures)
    }

    private func featureSet(
        includingCandidateFeatures: Bool
    ) -> Set<MLXModelOptimizationFeature> {
        var features = implementedFeatureSet()
        if includingCandidateFeatures {
            features.formUnion(candidateFeatureSet())
        }
        return features
    }

    private func implementedFeatureSet() -> Set<MLXModelOptimizationFeature> {
        var features = Set<MLXModelOptimizationFeature>()
        insertImplementedQuantizationFeatures(into: &features)
        insertImplementedRuntimeFeatures(into: &features)
        return features
    }

    private func candidateFeatureSet() -> Set<MLXModelOptimizationFeature> {
        var features = Set<MLXModelOptimizationFeature>()
        if isOQQuantized {
            features.insert(.oQQuantization)
        }
        if supportsNativeMTP {
            features.insert(.nativeMTP)
        }
        if supportsVLMMTP {
            features.insert(.vlmMTP)
        }
        if supportsVLMMTPDrafter {
            features.insert(.vlmMTPDrafter)
        }
        if supportsSpeculativePrefill {
            features.insert(.speculativePrefill)
        }
        if supportsDFlash {
            features.insert(.dFlash)
        }
        return features
    }

    private func insertImplementedQuantizationFeatures(
        into features: inout Set<MLXModelOptimizationFeature>
    ) {
        if requiresFP8ScaleDequantization {
            features.insert(.fp8ScaleDequantization)
        }
    }

    private func insertImplementedRuntimeFeatures(
        into features: inout Set<MLXModelOptimizationFeature>
    ) {
        if nativeMTPRuntimeSupported {
            features.insert(.nativeMTP)
        }
        if supportsIndexCache {
            features.insert(.indexCache)
        }
        if supportsTurboQuantKV {
            features.insert(.turboQuantKV)
        }
        if promptCacheReuseAlignment == .prefillStep {
            features.insert(.prefillStepPromptCacheReuse)
        }
    }
}
