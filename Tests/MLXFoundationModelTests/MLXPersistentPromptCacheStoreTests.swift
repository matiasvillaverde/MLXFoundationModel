import Foundation
@testable import MLXLocalModels
import Testing

@Suite("MLX persistent prompt cache store")
struct MLXPersistentPromptCacheStoreTests {
    @Test("uses standalone MLXFoundationModel cache namespace")
    func usesStandaloneMLXFoundationModelCacheNamespace() {
        let rootComponents = Set(MLXPersistentPromptCacheStore.rootURL().pathComponents)
        let blockComponents = Set(MLXPersistentPromptCacheBlockStore.rootURL().pathComponents)

        #expect(rootComponents.contains("MLXFoundationModel"))
        #expect(blockComponents.contains("MLXFoundationModel"))
        #expect(!rootComponents.contains("PatagoniaAppStore"))
        #expect(!blockComponents.contains("PatagoniaAppStore"))
        #expect(MLXPersistentPromptCacheStore.metadataKey.hasPrefix("org.mlxfoundationmodel."))
        #expect(!MLXPersistentPromptCacheStore.metadataKey.hasPrefix("patagonia."))
    }

    @Test("enforces total cache budget by pruning least-recent files")
    func enforcesTotalCacheBudgetByPruningLeastRecentFiles() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let old = try Self.writeCacheFile(named: "old.safetensors", bytes: 10, age: 3, root: root)
        let middle = try Self.writeCacheFile(named: "middle.safetensors", bytes: 20, age: 2, root: root)
        let recent = try Self.writeCacheFile(named: "recent.safetensors", bytes: 30, age: 1, root: root)

        try MLXPersistentPromptCacheStore.enforceBudget(rootURL: root, limitBytes: 50)

        #expect(!FileManager.default.fileExists(atPath: old.path))
        #expect(FileManager.default.fileExists(atPath: middle.path))
        #expect(FileManager.default.fileExists(atPath: recent.path))
    }

    @Test("keeps protected cache file while pruning around it")
    func keepsProtectedCacheFileWhilePruningAroundIt() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let protected = try Self.writeCacheFile(named: "protected.safetensors", bytes: 40, age: 3, root: root)
        let stale = try Self.writeCacheFile(named: "stale.safetensors", bytes: 20, age: 2, root: root)
        let newest = try Self.writeCacheFile(named: "newest.safetensors", bytes: 10, age: 1, root: root)

        try MLXPersistentPromptCacheStore.enforceBudget(
            rootURL: root,
            limitBytes: 40,
            protectedURL: protected
        )

        #expect(FileManager.default.fileExists(atPath: protected.path))
        #expect(!FileManager.default.fileExists(atPath: stale.path))
        #expect(!FileManager.default.fileExists(atPath: newest.path))
    }

    @Test("zero budget removes unprotected cache files only")
    func zeroBudgetRemovesUnprotectedCacheFilesOnly() throws {
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }

        let protected = try Self.writeCacheFile(named: "protected.safetensors", bytes: 10, age: 2, root: root)
        let unprotected = try Self.writeCacheFile(
            named: "unprotected.safetensors",
            bytes: 10,
            age: 1,
            root: root
        )
        let ignored = try Self.writeCacheFile(named: "notes.txt", bytes: 10, age: 3, root: root)

        try MLXPersistentPromptCacheStore.enforceBudget(
            rootURL: root,
            limitBytes: 0,
            protectedURL: protected
        )

        #expect(FileManager.default.fileExists(atPath: protected.path))
        #expect(!FileManager.default.fileExists(atPath: unprotected.path))
        #expect(FileManager.default.fileExists(atPath: ignored.path))
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: root,
            withIntermediateDirectories: true
        )
        return root
    }

    private static func writeCacheFile(
        named name: String,
        bytes: Int,
        age: TimeInterval,
        root: URL
    ) throws -> URL {
        var url = root.appendingPathComponent(name)
        try Data(repeating: 0, count: bytes).write(to: url)
        var values = URLResourceValues()
        let date = Date(timeIntervalSinceNow: -age)
        values.contentAccessDate = date
        values.contentModificationDate = date
        try url.setResourceValues(values)
        return url
    }
}
