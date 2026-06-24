import Foundation
@testable import MLXLocalModels
import Testing

@Suite("LoRA data loading")
struct LoRADataTests {
    @Test("directory lookup prefers JSONL over text")
    func directoryLookupPrefersJSONL() throws {
        let directory = try Self.temporaryDirectory()
        try Self.write("plain\n", to: directory.appending(component: "train.txt"))
        try Self.write("{\"text\":\"json\"}\n", to: directory.appending(component: "train.jsonl"))

        let samples = try loadLoRAData(directory: directory, name: "train")

        #expect(samples == ["json"])
    }

    @Test("loads non-empty text lines without trimming content")
    func loadsTextLines() throws {
        let url = try Self.temporaryDirectory().appending(component: "valid.txt")
        try Self.write(" first \n\nsecond\n", to: url)

        let samples = try loadLoRAData(url: url)

        #expect(samples == [" first ", "second"])
    }

    @Test("loads JSONL text records and skips non-object lines")
    func loadsJSONLines() throws {
        let url = try Self.temporaryDirectory().appending(component: "train.jsonl")
        try Self.write(
            [
                "{\"text\":\"one\"}",
                "not-json",
                "  {\"text\":\"two\"}",
                "{\"unused\":true}",
                ""
            ].joined(separator: "\n"),
            to: url
        )

        let samples = try loadLoRAData(url: url)

        #expect(samples == ["one", "two"])
    }

    @Test("reports missing named data files")
    func reportsMissingFiles() throws {
        let directory = try Self.temporaryDirectory()

        do {
            _ = try loadLoRAData(directory: directory, name: "missing")
            Issue.record("Expected missing data file error")
        } catch let error as LoRADataError {
            #expect(error == .fileNotFound(directory: directory, name: "missing"))
            #expect(error.localizedDescription.contains("missing"))
        }
    }

    @Test("reports unsupported direct file extensions")
    func reportsUnsupportedExtensions() throws {
        let url = try Self.temporaryDirectory().appending(component: "train.csv")
        try Self.write("text\n", to: url)

        do {
            _ = try loadLoRAData(url: url)
            Issue.record("Expected unsupported extension error")
        } catch let error as LoRADataError {
            #expect(error == .unsupportedFileExtension(url))
            #expect(error.localizedDescription.contains("csv"))
        }
    }

    private static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "MLXFoundationModel-LoRADataTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private static func write(_ contents: String, to url: URL) throws {
        try contents.write(to: url, atomically: true, encoding: .utf8)
    }
}
