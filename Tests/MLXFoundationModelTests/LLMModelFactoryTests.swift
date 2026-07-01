import Foundation
@testable import MLXLocalModels
import Testing

@Suite("LLM model factory")
struct LLMModelFactoryTests {
    @Test("registers grouped architecture aliases")
    func registersGroupedArchitectureAliases() {
        let registeredTypes = Set(LLMTypeRegistry.shared.registeredModelTypes())

        #expect(registeredTypes.isSuperset(of: [
            "mistral",
            "llama",
            "gemma3",
            "gemma3_text",
            "gemma4",
            "gemma4_text",
            "gemma4_assistant",
            "gemma4_unified_assistant",
            "glm4_moe_lite",
            "glm_moe_dsa",
            "cohere2",
            "phixtral",
            "phi-msft"
        ]))
    }

    @Test("generation token config falls back to base EOS ids")
    func generationTokenConfigFallsBackToBaseEOSIds() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let baseConfig = try Self.decodeBaseConfig(
            #"{"model_type":"qwen3","eos_token_id":[1,2]}"#
        )

        let result = LLMGenerationTokenConfig.load(
            baseConfig: baseConfig,
            modelDirectory: directory
        )

        #expect(result == LLMGenerationTokenConfig(eosTokenIds: [1, 2], suppressTokenIds: []))
    }

    @Test("generation token config overrides EOS ids and reads suppress tokens")
    func generationTokenConfigOverridesEOSIdsAndReadsSuppressTokens() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try #"{"eos_token_id":[7,8],"suppress_tokens":[3,5]}"#
            .write(
                to: directory.appending(component: "generation_config.json"),
                atomically: true,
                encoding: .utf8
            )
        let baseConfig = try Self.decodeBaseConfig(
            #"{"model_type":"qwen3","eos_token_id":[1,2]}"#
        )

        let result = LLMGenerationTokenConfig.load(
            baseConfig: baseConfig,
            modelDirectory: directory
        )

        #expect(result == LLMGenerationTokenConfig(eosTokenIds: [7, 8], suppressTokenIds: [3, 5]))
    }

    @Test("generation token config keeps base EOS ids when generation config only suppresses")
    func generationTokenConfigKeepsBaseEOSIdsWhenGenerationConfigOnlySuppresses() throws {
        let directory = try Self.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try #"{"suppress_tokens":9}"#
            .write(
                to: directory.appending(component: "generation_config.json"),
                atomically: true,
                encoding: .utf8
            )
        let baseConfig = try Self.decodeBaseConfig(
            #"{"model_type":"qwen3","eos_token_id":4}"#
        )

        let result = LLMGenerationTokenConfig.load(
            baseConfig: baseConfig,
            modelDirectory: directory
        )

        #expect(result == LLMGenerationTokenConfig(eosTokenIds: [4], suppressTokenIds: [9]))
    }

    private static func decodeBaseConfig(_ json: String) throws -> BaseConfiguration {
        try JSONDecoder.json5().decode(BaseConfiguration.self, from: Data(json.utf8))
    }

    private static func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "LLMModelFactoryTests-\(UUID().uuidString)", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
