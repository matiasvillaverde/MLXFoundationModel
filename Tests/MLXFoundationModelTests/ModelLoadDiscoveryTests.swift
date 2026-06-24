import Foundation
@testable import MLXLocalModels
import Testing

@Suite("Model load discovery")
struct ModelLoadDiscoveryTests {
    @Test("discovers safetensors recursively in deterministic path order")
    func discoversSafetensorsInDeterministicOrder() throws {
        let directory = try Self.temporaryDirectory()
        let nested = directory.appending(component: "nested")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        try Self.write("", to: directory.appending(component: "z.safetensors"))
        try Self.write("", to: nested.appending(component: "a.safetensors"))
        try Self.write("", to: directory.appending(component: "config.json"))

        let urls = try SafetensorFileDiscovery.safetensorURLs(in: directory)

        #expect(urls.map(\.lastPathComponent) == ["a.safetensors", "z.safetensors"])
    }

    @Test("matches safetensor extensions case-insensitively")
    func matchesSafetensorExtensionsCaseInsensitively() throws {
        let directory = try Self.temporaryDirectory()
        try Self.write("", to: directory.appending(component: "weights.SAFETENSORS"))

        let urls = try SafetensorFileDiscovery.safetensorURLs(in: directory)

        #expect(urls.map(\.lastPathComponent) == ["weights.SAFETENSORS"])
    }

    @Test("ignores directories that look like safetensor files")
    func ignoresDirectoriesThatLookLikeSafetensors() throws {
        let directory = try Self.temporaryDirectory()
        let fakeWeightDirectory = directory.appending(component: "fake.safetensors")
        try FileManager.default.createDirectory(at: fakeWeightDirectory, withIntermediateDirectories: true)
        try Self.write("", to: directory.appending(component: "real.safetensors"))

        let urls = try SafetensorFileDiscovery.safetensorURLs(in: directory)

        #expect(urls.map(\.lastPathComponent) == ["real.safetensors"])
    }

    @Test("reports directories that cannot be enumerated")
    func reportsDirectoriesThatCannotBeEnumerated() throws {
        let missingDirectory = FileManager.default.temporaryDirectory
            .appending(component: "MLXFoundationModel-Missing-\(UUID().uuidString)")

        do {
            _ = try SafetensorFileDiscovery.safetensorURLs(in: missingDirectory)
            Issue.record("Expected missing directory error")
        } catch let error as ModelLoadError {
            #expect(error == .cannotEnumerateWeights(missingDirectory))
            #expect(error.localizedDescription.contains(missingDirectory.lastPathComponent))
        }
    }

    private static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "MLXFoundationModel-ModelLoadDiscoveryTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func write(_ contents: String, to url: URL) throws {
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }
}
