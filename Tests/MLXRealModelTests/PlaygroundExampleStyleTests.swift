import MLXFoundationModel
import MLXFoundationModelExamples
import Testing

@Suite("FoundationModel playground example style")
struct PlaygroundExampleStyleTests {
    @Test("default examples use the model-native prompt style")
    func defaultExamplesUseModelNativePromptStyle() {
        let example = FoundationModelPlaygroundExamples.pointsOfInterestToolCalling

        #expect(example.resolvedStyle(modelDefault: .qwenXML) == .qwenXML)
        #expect(example.resolvedStyle(modelDefault: .harmony) == .harmony)
        #expect(example.resolvedStyle(modelDefault: .plain) == .chatML)
    }

    @Test("explicit non-default styles override model defaults")
    func explicitNonDefaultStylesOverrideModelDefaults() {
        let example = FoundationModelPlaygroundExample(
            id: "explicit",
            title: "Explicit",
            request: MLXBridgeRequest(messages: []),
            style: .plain,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 1)
        )

        #expect(example.resolvedStyle(modelDefault: .qwenXML) == .plain)
    }
}
