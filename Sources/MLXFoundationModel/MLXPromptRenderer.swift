import Foundation
import MLXLocalModels

/// Converts Foundation Models-style transcript data into text for local MLX models.
public enum MLXPromptRenderer {
    /// Render a bridge request into a prompt string and cache metadata.
    public static func render(
        _ request: MLXBridgeRequest,
        style: MLXPromptStyle
    ) -> MLXRenderedRequest {
        let rendererID = "mlx.\(style.codingValue).v1"
        let prompt = MLXPromptTemplateRenderer.render(request, style: style)
        let fingerprint = PromptCacheIdentity.stableFingerprint(for: rendererID)
        return MLXRenderedRequest(
            prompt: prompt,
            rendererID: rendererID,
            cacheFingerprint: fingerprint
        )
    }
}
