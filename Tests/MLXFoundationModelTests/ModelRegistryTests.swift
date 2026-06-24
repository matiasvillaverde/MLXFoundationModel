import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model configuration and registries")
struct ModelRegistryTests {
    @Test("keeps remote and local configuration identity")
    func keepsConfigurationIdentity() {
        let remote = ModelConfiguration(
            id: "owner/model",
            revision: "v2",
            tokenizerId: "owner/tokenizer",
            overrideTokenizer: "PreTrainedTokenizer",
            defaultPrompt: "Explain MLX.",
            extraEOSTokens: ["<eos>"],
            eosTokenIds: [1, 2],
            suppressTokenIds: [3]
        )

        #expect(remote.name == "owner/model")
        #expect(remote.tokenizerId == "owner/tokenizer")
        #expect(remote.overrideTokenizer == "PreTrainedTokenizer")
        #expect(remote.defaultPrompt == "Explain MLX.")
        #expect(remote.extraEOSTokens == ["<eos>"])
        #expect(remote.eosTokenIds == [1, 2])
        #expect(remote.suppressTokenIds == [3])

        let directory = URL(fileURLWithPath: "/tmp/owner/model", isDirectory: true)
        let local = ModelConfiguration(directory: directory)

        #expect(local.name == "owner/model")
        #expect(local.modelDirectory() == directory)
        #expect(local.extraEOSTokens == ["<end_of_turn>"])
    }

    @Test("compares ids with revisions and directories")
    func comparesIdentifiers() {
        let first = ModelConfiguration(id: "owner/model", revision: "v1")
        let same = ModelConfiguration(id: "owner/model", revision: "v1")
        let differentRevision = ModelConfiguration(id: "owner/model", revision: "v2")
        let directory = ModelConfiguration(
            directory: URL(fileURLWithPath: "/tmp/owner/model", isDirectory: true)
        )

        #expect(first == same)
        #expect(first != differentRevision)
        #expect(first != directory)
    }

    @Test("returns registered configuration or fallback")
    func returnsRegisteredConfigurationOrFallback() {
        let known = ModelConfiguration(
            id: "owner/known",
            defaultPrompt: "Known prompt",
            extraEOSTokens: ["<known>"]
        )
        let registry = AbstractModelRegistry(modelConfigurations: [known])

        #expect(registry.contains(id: "owner/known"))
        #expect(registry.configuration(id: "owner/known") == known)
        #expect(!registry.contains(id: "owner/missing"))

        let fallback = registry.configuration(id: "owner/missing")
        #expect(fallback.name == "owner/missing")
        #expect(fallback.defaultPrompt == "hello")
    }

    @Test("registers configuration replacements")
    func registersConfigurationReplacements() {
        let registry = AbstractModelRegistry(modelConfigurations: [
            ModelConfiguration(id: "owner/model", defaultPrompt: "old")
        ])
        let replacement = ModelConfiguration(id: "owner/model", defaultPrompt: "new")
        let other = ModelConfiguration(id: "owner/other")

        registry.register(configurations: [replacement, other])

        #expect(registry.configuration(id: "owner/model") == replacement)
        #expect(Set(registry.models.map(\.name)) == ["owner/model", "owner/other"])
    }

    @Test("creates models from registered type creators")
    func createsModelsFromRegisteredTypes() throws {
        let registry = ModelTypeRegistry(creators: [
            "echo": { _ in MLXEchoBatchLanguageModel() }
        ])
        let configuration = URL(fileURLWithPath: "/tmp/config.json")

        let model = try registry.createModel(configuration: configuration, modelType: "echo")

        #expect(model is MLXEchoBatchLanguageModel)
        #expect(registry.registeredModelTypes() == ["echo"])

        registry.registerModelType("second") { _ in MLXEchoBatchLanguageModel() }
        #expect(registry.registeredModelTypes() == ["echo", "second"])
    }

    @Test("throws the requested model type when missing")
    func throwsRequestedModelTypeWhenMissing() {
        let registry = ModelTypeRegistry()
        let configuration = URL(fileURLWithPath: "/tmp/config.json")

        do {
            _ = try registry.createModel(configuration: configuration, modelType: "missing")
            Issue.record("Expected unsupported model type")
        } catch ModelFactoryError.unsupportedModelType(let modelType) {
            #expect(modelType == "missing")
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }

    @Test("serializes concurrent registry writes")
    func serializesConcurrentRegistryWrites() async {
        let registry = AbstractModelRegistry()

        await withTaskGroup(of: Void.self) { group in
            for index in 0 ..< 50 {
                group.addTask {
                    let configuration = ModelConfiguration(id: "owner/model-\(index)")
                    registry.register(configurations: [configuration])
                    _ = registry.configuration(id: configuration.name)
                }
            }
        }

        #expect(registry.models.count == 50)
    }
}
