import Foundation

/// Optimization features that may be detected from an MLX model profile.
public enum MLXModelOptimizationFeature: String, Codable, CaseIterable, Hashable, Sendable {
    case dFlash = "d_flash"
    case fp8ScaleDequantization = "fp8_scale_dequantization"
    case indexCache = "index_cache"
    case nativeMTP = "native_mtp"
    case oQQuantization = "oq_quantization"
    case prefillStepPromptCacheReuse = "prefill_step_prompt_cache_reuse"
    case speculativePrefill = "speculative_prefill"
    case turboQuantKV = "turbo_quant_kv"
    case vlmMTP = "vlm_mtp"
    case vlmMTPDrafter = "vlm_mtp_drafter"
}
