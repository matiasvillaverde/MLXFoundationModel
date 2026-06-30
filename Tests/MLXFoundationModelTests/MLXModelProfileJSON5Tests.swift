import Foundation
@testable import MLXFoundationModel
import Testing

@Suite("MLX model profile JSON5 loading")
struct MLXModelProfileJSON5Tests {
    @Test("loads profiles from JSON5 model configs")
    func loadsProfilesFromJSON5ModelConfigs() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("MLXModelProfileJSON5-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let configText = """
        {
            "model_type": "mamba2",
            "architectures": ["Mamba2ForCausalLM"],
            "vocab_size": 1024,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "hidden_size": 16,
            "head_dim": 4,
            "time_step_limit": [0.0, Infinity]
        }
        """
        try Data(configText.utf8).write(to: directory.appendingPathComponent("config.json"))

        let model = try MLXModel.profiled(id: "mamba2-json5-fixture", location: directory)
        let profile = try #require(model.profile)

        #expect(profile.modelType == "mamba2")
        #expect(profile.architectures == ["Mamba2ForCausalLM"])
        #expect(profile.vocabularySize == 1_024)
    }
}
