import Foundation

internal enum MLXPersistentPromptCacheBudgetEnforcer {
    private struct BudgetCandidate {
        enum Kind {
            case snapshot(URL)
            case block(MLXPersistentPromptCacheBlockRecord, URL)
        }

        let byteCount: Int
        let lastAccess: Date
        let stablePath: String
        let kind: Kind

        func isProtected(
            snapshotPath: String?,
            storageHashes: Set<String>,
            protectionEnabled: Bool
        ) -> Bool {
            guard protectionEnabled else {
                return false
            }
            switch kind {
            case .snapshot:
                return stablePath == snapshotPath

            case .block(let record, _):
                return storageHashes.contains(record.storageHash)
            }
        }

        func remove(fileManager: FileManager) {
            switch kind {
            case .snapshot(let url):
                try? fileManager.removeItem(at: url)

            case .block(let record, let rootURL):
                MLXPersistentPromptCacheBlockStore.removeRecord(
                    record,
                    rootURL: rootURL,
                    fileManager: fileManager
                )
            }
        }
    }

    internal static func enforceAll(
        limitBytes: Int,
        protectedSnapshotURL: URL? = nil,
        protectedStorageHashes: Set<String> = [],
        snapshotRootURL: URL = MLXPersistentPromptCacheStore.rootURL(),
        blockRootURL: URL = MLXPersistentPromptCacheBlockStore.rootURL(),
        segmentRootURL: URL = MLXPersistentPromptCacheSegmentStore.rootURL(),
        fileManager: FileManager = .default
    ) throws {
        let normalizedProtectedHashes = Set(protectedStorageHashes.map {
            MLXPersistentPromptCacheBlockStore.normalizedHash($0)
        })
        let candidates = try budgetCandidates(
            snapshotRootURL: snapshotRootURL,
            blockRootURL: blockRootURL,
            segmentRootURL: segmentRootURL,
            fileManager: fileManager
        )
        try enforce(
            candidates: candidates,
            limitBytes: limitBytes,
            protectedSnapshotURL: protectedSnapshotURL,
            protectedStorageHashes: normalizedProtectedHashes,
            fileManager: fileManager
        )
    }

    internal static func enforceAllBeforeInsert(
        limitBytes: Int,
        incomingByteCount: Int,
        protectedSnapshotURL: URL? = nil,
        protectedStorageHashes: Set<String> = [],
        snapshotRootURL: URL = MLXPersistentPromptCacheStore.rootURL(),
        blockRootURL: URL = MLXPersistentPromptCacheBlockStore.rootURL(),
        segmentRootURL: URL = MLXPersistentPromptCacheSegmentStore.rootURL(),
        fileManager: FileManager = .default
    ) throws {
        try enforceAll(
            limitBytes: max(0, limitBytes - max(0, incomingByteCount)),
            protectedSnapshotURL: protectedSnapshotURL,
            protectedStorageHashes: protectedStorageHashes,
            snapshotRootURL: snapshotRootURL,
            blockRootURL: blockRootURL,
            segmentRootURL: segmentRootURL,
            fileManager: fileManager
        )
    }

    private static func budgetCandidates(
        snapshotRootURL: URL,
        blockRootURL: URL,
        segmentRootURL: URL,
        fileManager: FileManager
    ) throws -> [BudgetCandidate] {
        try snapshotCandidates(rootURL: snapshotRootURL, fileManager: fileManager) +
            blockCandidates(rootURL: blockRootURL, fileManager: fileManager) +
            blockCandidates(rootURL: segmentRootURL, fileManager: fileManager)
    }

    private static func enforce(
        candidates: [BudgetCandidate],
        limitBytes: Int,
        protectedSnapshotURL: URL?,
        protectedStorageHashes: Set<String>,
        fileManager: FileManager
    ) throws {
        let protectedSnapshotPath = protectedSnapshotURL?.standardizedFileURL.path
        var totalBytes = candidates.reduce(0) { $0 + $1.byteCount }
        for candidate in sortedByAccess(candidates) where totalBytes > limitBytes {
            guard !candidate.isProtected(
                snapshotPath: protectedSnapshotPath,
                storageHashes: protectedStorageHashes,
                protectionEnabled: limitBytes > 0
            ) else {
                continue
            }
            candidate.remove(fileManager: fileManager)
            totalBytes -= candidate.byteCount
            MLXGenerationDiagnostics.recordPromptCacheEviction()
        }
    }

    private static func snapshotCandidates(
        rootURL: URL,
        fileManager: FileManager
    ) throws -> [BudgetCandidate] {
        try MLXPersistentPromptCacheStore.cacheFiles(
            rootURL: rootURL,
            fileManager: fileManager
        ).map { file in
            BudgetCandidate(
                byteCount: file.byteCount,
                lastAccess: file.lastAccessDate,
                stablePath: file.url.standardizedFileURL.path,
                kind: .snapshot(file.url)
            )
        }
    }

    private static func blockCandidates(
        rootURL: URL,
        fileManager: FileManager
    ) throws -> [BudgetCandidate] {
        try MLXPersistentPromptCacheBlockStore.scan(
            rootURL: rootURL,
            removeStaleFiles: true,
            fileManager: fileManager
        ).map { record in
            let url = MLXPersistentPromptCacheBlockStore.dataURL(
                for: record,
                rootURL: rootURL
            )
            return BudgetCandidate(
                byteCount: record.byteCount,
                lastAccess: record.lastAccess,
                stablePath: url.standardizedFileURL.path,
                kind: .block(record, rootURL)
            )
        }
    }

    private static func sortedByAccess(
        _ candidates: [BudgetCandidate]
    ) -> [BudgetCandidate] {
        candidates.sorted { lhs, rhs in
            if lhs.lastAccess == rhs.lastAccess {
                return lhs.stablePath < rhs.stablePath
            }
            return lhs.lastAccess < rhs.lastAccess
        }
    }
}
