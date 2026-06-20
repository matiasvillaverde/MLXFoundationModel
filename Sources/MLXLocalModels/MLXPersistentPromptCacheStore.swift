import Foundation

internal enum MLXPersistentPromptCacheStore {
    internal static let metadataKey = "org.mlxfoundationmodel.prompt_cache.envelope.v1"
    private static let cacheDirectoryName = "PromptCache"
    private static let packageCacheDirectoryName = "MLXFoundationModel"

    internal static func rootURL() -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first ??
            FileManager.default.temporaryDirectory
        return root
            .appendingPathComponent(packageCacheDirectoryName, isDirectory: true)
            .appendingPathComponent(cacheDirectoryName, isDirectory: true)
    }

    internal static func url(for configuration: ProviderConfiguration) -> URL {
        rootURL().appendingPathComponent(filename(for: configuration))
    }

    internal static func filename(for configuration: ProviderConfiguration) -> String {
        let fingerprintSeed = [
            configuration.modelName,
            configuration.location.standardizedFileURL.path,
            String(configuration.compute.contextSize)
        ].joined(separator: "|")
        return "\(PromptCacheIdentity.stableFingerprint(for: fingerprintSeed)).safetensors"
    }

    internal static func enforceBudget(
        rootURL: URL = rootURL(),
        limitBytes: Int,
        protectedURL: URL? = nil,
        fileManager: FileManager = .default
    ) throws {
        guard limitBytes > 0 else {
            try removeAllCacheFiles(
                rootURL: rootURL,
                protectedURL: protectedURL,
                fileManager: fileManager
            )
            return
        }

        let protectedPath = protectedURL?.standardizedFileURL.path
        var files = try cacheFiles(rootURL: rootURL, fileManager: fileManager)
        var totalBytes = files.reduce(0) { $0 + $1.byteCount }
        files.sort { lhs, rhs in
            if lhs.lastAccessDate == rhs.lastAccessDate {
                return lhs.url.path < rhs.url.path
            }
            return lhs.lastAccessDate < rhs.lastAccessDate
        }

        for file in files where totalBytes > limitBytes {
            guard file.url.standardizedFileURL.path != protectedPath else {
                continue
            }
            try? fileManager.removeItem(at: file.url)
            totalBytes -= file.byteCount
        }
    }

    private static func removeAllCacheFiles(
        rootURL: URL,
        protectedURL: URL?,
        fileManager: FileManager
    ) throws {
        let protectedPath = protectedURL?.standardizedFileURL.path
        for file in try cacheFiles(rootURL: rootURL, fileManager: fileManager) {
            guard file.url.standardizedFileURL.path != protectedPath else {
                continue
            }
            try? fileManager.removeItem(at: file.url)
        }
    }

    internal static func cacheFiles(
        rootURL: URL,
        fileManager: FileManager
    ) throws -> [CacheFile] {
        guard fileManager.fileExists(atPath: rootURL.path) else {
            return []
        }
        let urls = try fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [
                .contentAccessDateKey,
                .contentModificationDateKey,
                .fileSizeKey,
                .isRegularFileKey
            ],
            options: [.skipsHiddenFiles]
        )
        return urls.compactMap { url in
            guard url.pathExtension == "safetensors" else {
                return nil
            }
            return CacheFile(url: url)
        }
    }

    internal struct CacheFile {
        let url: URL
        let byteCount: Int
        let lastAccessDate: Date

        init?(url: URL) {
            guard let values = try? url.resourceValues(forKeys: [
                .contentAccessDateKey,
                .contentModificationDateKey,
                .fileSizeKey,
                .isRegularFileKey
            ]),
                values.isRegularFile == true,
                let byteCount = values.fileSize
            else {
                return nil
            }
            self.url = url
            self.byteCount = byteCount
            self.lastAccessDate = values.contentAccessDate ??
                values.contentModificationDate ??
                .distantPast
        }
    }
}
