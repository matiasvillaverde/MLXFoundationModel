import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model runtime prompt cache preferences")
struct ModelRuntimePromptCachePreferenceTests {
    @Test("decodes missing prompt cache alignment as exact")
    func decodesMissingPromptCacheAlignmentAsExact() throws {
        let data = Data(Self.legacyRuntimeJSON.utf8)

        let preferences = try JSONDecoder().decode(ModelRuntimePreferences.self, from: data)

        #expect(preferences.promptCacheReuseAlignment == .exact)
    }

    @Test("encodes prompt cache alignment preference")
    func encodesPromptCacheAlignmentPreference() throws {
        let preferences = ModelRuntimePreferences(promptCacheReuseAlignment: .prefillStep)

        let data = try JSONEncoder().encode(preferences)
        let decoded = try JSONDecoder().decode(ModelRuntimePreferences.self, from: data)

        #expect(decoded.promptCacheReuseAlignment == .prefillStep)
    }

    @Test("applies required prefill-step prompt cache alignment")
    func appliesRequiredPrefillStepPromptCacheAlignment() {
        let preferences = ModelRuntimePreferences(
            promptCachePolicy: .persistent,
            promptCacheReuseAlignment: .exact
        )

        let promoted = preferences.applyingRequiredPromptCacheReuseAlignment(.prefillStep)

        #expect(promoted.promptCacheReuseAlignment == .prefillStep)
        #expect(promoted.promptCachePolicy == .persistent)
    }

    private static let legacyRuntimeJSON = """
    {
        "residencyPreference": "warm",
        "isPinned": false,
        "idleTTLSeconds": 300,
        "promptCachePolicy": "persistent",
        "promptCacheByteLimit": 134217728,
        "speculativeDecodingMode": "off",
        "speculativeDraftTokens": 2
    }
    """
}
