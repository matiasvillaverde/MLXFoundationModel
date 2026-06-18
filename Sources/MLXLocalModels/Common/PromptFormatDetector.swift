import Foundation

internal enum PromptFormatDetector {
    private static let knownPromptMarkers: [String] = [
        "<|turn>",
        "<|im_start|>",
        "<start_of_turn>",
        "<|begin_of_text|>",
        "[INST]",
        "<s>[INST]",
        "<bos><|turn>"
    ]

    internal static func isPreformatted(_ prompt: String) -> Bool {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return knownPromptMarkers.contains { marker in
            trimmed.hasPrefix(marker) || trimmed.contains("\n\(marker)")
        }
    }
}
