import Foundation
import MLXFoundationModel
import Testing

@Suite(
    "MLX real-model constrained decoding",
    .serialized,
    .disabled(
        if: !MLXRealModelEnvironment.isEnabled,
        "Set MLX_RUN_REAL_MODEL_TESTS=1 and download models with scripts/download-test-models.sh"
    )
)
struct MLXRealModelConstrainedDecodingTests {
    @Test("Qwen3 follows rendered JSON response constraints")
    func qwen3FollowsRenderedJSONResponseConstraints() async throws {
        let models = try MLXRealModelCatalog.load()
        let model = try MLXRealModelHarness.requireModel("qwen3-0.6b-4bit", in: models)
        let request = MLXBridgeRequest(
            messages: [
                MLXBridgeMessage(
                    role: .user,
                    content: "/no_think\nReturn a forecast for Berlin with celsius 21."
                )
            ],
            instructions: "You are a JSON encoder. Do not think aloud. Return JSON only.",
            responseConstraint: MLXBridgeResponseConstraint(
                jsonSchema: #"{"type":"object","required":["city","celsius"]}"#
            )
        )
        let rendered = MLXPromptRenderer.render(request, style: .chatML)
        let result = try await MLXRealModelHarness.run(
            model: model,
            prompt: rendered.prompt,
            sampling: .deterministic,
            limits: ResourceLimits(maxTokens: 160, maxTime: .seconds(120), reusePromptCache: false)
        )

        MLXRealModelHarness.verifyGenerated(result)
        let json = try Self.extractJSONObject(from: result.text)
        #expect(json["city"] as? String == "Berlin")
        #expect(json["celsius"] != nil)
    }

    private static func extractJSONObject(from text: String) throws -> [String: Any] {
        let jsonText = try #require(MLXJSONTextExtractor.firstJSONObject(in: text))
        let data = try #require(jsonText.data(using: .utf8))
        return try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
    }
}
