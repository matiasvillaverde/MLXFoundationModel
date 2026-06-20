import CryptoKit
import Foundation

internal enum MLXPersistentPromptCacheBlockPayloadKind: String, Codable, Sendable {
    case block
    case prefixSnapshot
    case generic
    case compactedRotatingTip
}

internal enum MLXPersistentPromptCacheHotPayloadPromotionPolicy: Sendable {
    case evictIfNeeded
    case skipIfWouldEvict
}

internal struct MLXPersistentPromptCacheBlockRecord: Codable, Equatable, Sendable {
    private enum CodingKeys: String, CodingKey {
        case blockHash
        case storageHash
        case payloadKind
        case blockSize
        case tokenCount
        case byteCount
        case signature
        case createdAt
        case lastAccess
    }

    let blockHash: String
    let storageHash: String
    let payloadKind: MLXPersistentPromptCacheBlockPayloadKind
    let blockSize: Int
    let tokenCount: Int
    let byteCount: Int
    let signature: PromptCacheSignature
    let createdAt: Date
    let lastAccess: Date

    internal init(
        blockHash: String,
        storageHash: String? = nil,
        payloadKind: MLXPersistentPromptCacheBlockPayloadKind = .generic,
        blockSize: Int = 256,
        tokenCount: Int,
        byteCount: Int,
        signature: PromptCacheSignature,
        createdAt: Date,
        lastAccess: Date
    ) {
        let normalizedBlockHash = MLXPersistentPromptCacheBlockStore.normalizedHash(blockHash)
        self.blockHash = normalizedBlockHash
        self.storageHash = storageHash.map(MLXPersistentPromptCacheBlockStore.normalizedHash)
            ?? MLXPersistentPromptCacheBlockStore.storageHash(
                blockHash: normalizedBlockHash,
                signature: signature,
                blockSize: max(1, blockSize),
                payloadKind: payloadKind
            )
        self.payloadKind = payloadKind
        self.blockSize = max(1, blockSize)
        self.tokenCount = tokenCount
        self.byteCount = max(0, byteCount)
        self.signature = signature
        self.createdAt = createdAt
        self.lastAccess = lastAccess
    }

    internal init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let blockHash = try container.decode(String.self, forKey: .blockHash)
        let signature = try container.decode(PromptCacheSignature.self, forKey: .signature)
        let blockSize = try container.decode(Int.self, forKey: .blockSize)
        self.init(
            blockHash: blockHash,
            storageHash: try container.decodeIfPresent(String.self, forKey: .storageHash) ?? blockHash,
            payloadKind: try container.decodeIfPresent(
                MLXPersistentPromptCacheBlockPayloadKind.self,
                forKey: .payloadKind
            ) ?? .generic,
            blockSize: blockSize,
            tokenCount: try container.decode(Int.self, forKey: .tokenCount),
            byteCount: try container.decode(Int.self, forKey: .byteCount),
            signature: signature,
            createdAt: try container.decode(Date.self, forKey: .createdAt),
            lastAccess: try container.decode(Date.self, forKey: .lastAccess)
        )
    }

    internal func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(blockHash, forKey: .blockHash)
        try container.encode(storageHash, forKey: .storageHash)
        try container.encode(payloadKind, forKey: .payloadKind)
        try container.encode(blockSize, forKey: .blockSize)
        try container.encode(tokenCount, forKey: .tokenCount)
        try container.encode(byteCount, forKey: .byteCount)
        try container.encode(signature, forKey: .signature)
        try container.encode(createdAt, forKey: .createdAt)
        try container.encode(lastAccess, forKey: .lastAccess)
    }

    internal func accessed(at date: Date) -> Self {
        Self(
            blockHash: blockHash,
            storageHash: storageHash,
            payloadKind: payloadKind,
            blockSize: blockSize,
            tokenCount: tokenCount,
            byteCount: byteCount,
            signature: signature,
            createdAt: createdAt,
            lastAccess: date
        )
    }
}

internal struct MLXPersistentPromptCachePendingBlock: Sendable {
    let blockHash: String
    let blockSize: Int
    let tokenCount: Int
    let signature: PromptCacheSignature
    let payload: Data
    let payloadKind: MLXPersistentPromptCacheBlockPayloadKind

    internal init(
        blockHash: String,
        blockSize: Int = 256,
        tokenCount: Int,
        signature: PromptCacheSignature,
        payload: Data,
        payloadKind: MLXPersistentPromptCacheBlockPayloadKind = .generic
    ) {
        self.blockHash = MLXPersistentPromptCacheBlockStore.normalizedHash(blockHash)
        self.blockSize = max(1, blockSize)
        self.tokenCount = tokenCount
        self.signature = signature
        self.payload = payload
        self.payloadKind = payloadKind
    }

    internal var byteCount: Int {
        payload.count
    }
}

internal struct MLXPersistentPromptCacheTipDescriptor: Sendable {
    let blockHash: String
    let blockSize: Int
    let signature: PromptCacheSignature
    let payloadKind: MLXPersistentPromptCacheBlockPayloadKind
    let rootURL: URL

    internal init(
        blockHash: String,
        blockSize: Int,
        signature: PromptCacheSignature,
        payloadKind: MLXPersistentPromptCacheBlockPayloadKind = .block,
        rootURL: URL
    ) {
        self.blockHash = MLXPersistentPromptCacheBlockStore.normalizedHash(blockHash)
        self.blockSize = max(1, blockSize)
        self.signature = signature
        self.payloadKind = payloadKind
        self.rootURL = rootURL
    }

    var storageHash: String {
        MLXPersistentPromptCacheBlockStore.storageHash(
            blockHash: blockHash,
            signature: signature,
            blockSize: blockSize,
            payloadKind: payloadKind
        )
    }

    var lineageKey: String {
        rootURL.standardizedFileURL.path + "#" + storageHash
    }
}

private final class MLXPersistentPromptCacheTipLineageCompactor: @unchecked Sendable {
    private let lock = NSLock()
    private let maxEntries: Int
    private var lineage: [String: MLXPersistentPromptCacheTipDescriptor] = [:]

    init(maxEntries: Int) {
        self.maxEntries = max(1, maxEntries)
        lock.name = "org.mlxfoundationmodel.persistent-cache-tip-lineage"
    }

    func recordExtension(
        previousTip: MLXPersistentPromptCacheTipDescriptor,
        newTip: MLXPersistentPromptCacheTipDescriptor
    ) -> MLXPersistentPromptCacheTipDescriptor? {
        lock.lock()
        defer { lock.unlock() }
        let superseded = lineage.removeValue(forKey: previousTip.lineageKey)
        lineage[newTip.lineageKey] = previousTip
        if lineage.count > maxEntries {
            lineage.removeAll(keepingCapacity: true)
        }
        return superseded
    }

    func removeAll() {
        lock.lock()
        lineage.removeAll(keepingCapacity: true)
        lock.unlock()
    }
}

internal struct MLXPersistentPromptCachePrefixHit: Equatable, Sendable {
    let records: [MLXPersistentPromptCacheBlockRecord]
    let dataURLs: [URL]
    let matchedBlockCount: Int
    let cachedTokenCount: Int
    let nextMissingBlockHash: String?
}

internal struct MLXPersistentPromptCacheSnapshotHit: Equatable, Sendable {
    let record: MLXPersistentPromptCacheBlockRecord
    let dataURL: URL
    let matchedBlockCount: Int
    let cachedTokenCount: Int
}

internal enum MLXPersistentPromptCacheBlockStore {
    private static let cacheDirectoryName = "PromptCacheBlocks"
    private static let packageCacheDirectoryName = "MLXFoundationModel"
    private static let dataExtension = "safetensors"
    private static let metadataExtension = "json"
    private static let hotCacheDefaultByteLimit = 67_108_864
    private static let hotPayloadCache = MLXPersistentPromptCacheHotPayloadCache(
        maxBytes: hotCacheDefaultByteLimit
    )
    private static let tipLineageCompactor = MLXPersistentPromptCacheTipLineageCompactor(
        maxEntries: 4_096
    )
    private static let compactedRotatingTipPayload = Data(
        "MLXFoundationModel.compacted-rotating-tip.v1".utf8
    )

    internal static func rootURL() -> URL {
        let root = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first ??
            FileManager.default.temporaryDirectory
        return root
            .appendingPathComponent(packageCacheDirectoryName, isDirectory: true)
            .appendingPathComponent(cacheDirectoryName, isDirectory: true)
    }

    internal static func normalizedHash(_ blockHash: String) -> String {
        blockHash
            .lowercased()
            .filter { character in
                ("0" ... "9").contains(character) || ("a" ... "f").contains(character)
            }
    }

    internal static func storageHash(
        blockHash: String,
        signature: PromptCacheSignature,
        blockSize: Int,
        payloadKind: MLXPersistentPromptCacheBlockPayloadKind = .generic
    ) -> String {
        var hasher = SHA256()
        hasher.update(data: Data(normalizedHash(blockHash).utf8))
        var blockSizeValue = UInt64(max(1, blockSize)).littleEndian
        withUnsafeBytes(of: &blockSizeValue) { hasher.update(bufferPointer: $0) }
        hasher.update(data: Data(payloadKind.rawValue.utf8))
        if let signatureData = try? canonicalEncoder.encode(signature) {
            hasher.update(data: signatureData)
        }
        return hasher.finalize().hexString
    }

    internal static func dataURL(for blockHash: String, rootURL: URL = rootURL()) -> URL {
        blockDirectory(for: blockHash, rootURL: rootURL)
            .appendingPathComponent(normalizedHash(blockHash))
            .appendingPathExtension(dataExtension)
    }

    internal static func metadataURL(for blockHash: String, rootURL: URL = rootURL()) -> URL {
        blockDirectory(for: blockHash, rootURL: rootURL)
            .appendingPathComponent(normalizedHash(blockHash))
            .appendingPathExtension(metadataExtension)
    }

    internal static func dataURL(
        for record: MLXPersistentPromptCacheBlockRecord,
        rootURL: URL = rootURL()
    ) -> URL {
        dataURL(
            forStorageHash: record.storageHash,
            logicalBlockHash: record.blockHash,
            rootURL: rootURL
        )
    }

    internal static func metadataURL(
        for record: MLXPersistentPromptCacheBlockRecord,
        rootURL: URL = rootURL()
    ) -> URL {
        metadataURL(
            forStorageHash: record.storageHash,
            logicalBlockHash: record.blockHash,
            rootURL: rootURL
        )
    }

    internal static func configureHotCache(limitBytes: Int) {
        let evictedCount = hotPayloadCache.configure(maxBytes: limitBytes)
        recordHotCacheEvictions(evictedCount)
    }

    internal static func clearHotCache() {
        hotPayloadCache.removeAll()
    }

    internal static func clearTipLineage() {
        tipLineageCompactor.removeAll()
    }

    internal static func hotCacheSnapshot() -> MLXPersistentPromptCacheHotPayloadSnapshot {
        hotPayloadCache.snapshot()
    }

    internal static func cappedToHotCacheCapacity(
        _ hit: MLXPersistentPromptCachePrefixHit
    ) -> MLXPersistentPromptCachePrefixHit {
        let snapshot = hotPayloadCache.snapshot()
        let hotKeys = Set(snapshot.keys)
        var availableBytes = snapshot.availableBytes
        var records: [MLXPersistentPromptCacheBlockRecord] = []
        var dataURLs: [URL] = []

        for (record, dataURL) in zip(hit.records, hit.dataURLs) {
            let key = hotPayloadKey(for: dataURL)
            guard hotKeys.contains(key) || reservePayload(record, dataURL: dataURL, from: &availableBytes) else {
                break
            }
            records.append(record)
            dataURLs.append(dataURL)
        }

        let cachedTokenCount = records.reduce(0) { total, record in
            total + record.tokenCount
        }
        return MLXPersistentPromptCachePrefixHit(
            records: records,
            dataURLs: dataURLs,
            matchedBlockCount: records.count,
            cachedTokenCount: cachedTokenCount,
            nextMissingBlockHash: nextMissingBlockHash(original: hit, selectedCount: records.count)
        )
    }

    internal static func cappedToReusePolicy(
        _ hit: MLXPersistentPromptCachePrefixHit,
        requestTokenCount: Int,
        reusePolicy: PromptCacheReusePolicy
    ) -> MLXPersistentPromptCachePrefixHit? {
        let blockSize = hit.records.first?.blockSize ?? 1
        let reusableTokenCount = reusePolicy.persistentPrefixTokenCount(
            cachedTokenCount: hit.cachedTokenCount,
            requestTokenCount: requestTokenCount,
            blockSize: blockSize
        )
        guard reusableTokenCount > 0 else {
            return nil
        }
        guard reusableTokenCount < hit.cachedTokenCount else {
            return hit
        }

        var records: [MLXPersistentPromptCacheBlockRecord] = []
        var dataURLs: [URL] = []
        var selectedTokenCount = 0
        for (record, dataURL) in zip(hit.records, hit.dataURLs) {
            let nextTokenCount = selectedTokenCount + record.tokenCount
            guard nextTokenCount <= reusableTokenCount else {
                break
            }
            records.append(record)
            dataURLs.append(dataURL)
            selectedTokenCount = nextTokenCount
        }
        guard selectedTokenCount == reusableTokenCount else {
            return nil
        }

        return MLXPersistentPromptCachePrefixHit(
            records: records,
            dataURLs: dataURLs,
            matchedBlockCount: records.count,
            cachedTokenCount: selectedTokenCount,
            nextMissingBlockHash: nextMissingBlockHash(original: hit, selectedCount: records.count)
        )
    }

    internal static func loadPayload(
        at dataURL: URL,
        promotionPolicy: MLXPersistentPromptCacheHotPayloadPromotionPolicy = .evictIfNeeded
    ) throws -> Data {
        let key = hotPayloadKey(for: dataURL)
        if let data = hotPayloadCache.data(forKey: key) {
            MLXGenerationDiagnostics.recordPromptCacheSSDHotHit()
            return data
        }

        let data = try Data(contentsOf: dataURL)
        MLXGenerationDiagnostics.recordPromptCacheSSDDiskLoad()
        storeHotPayload(
            data,
            key: key,
            recordPromotion: true,
            promotionPolicy: promotionPolicy
        )
        return data
    }

    internal static func storedRecord(
        blockHash: String,
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        rootURL: URL = rootURL(),
        fileManager: FileManager = .default,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind]
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        try validRecord(
            blockHash: blockHash,
            signature: signature,
            blockSize: blockSize,
            rootURL: rootURL,
            fileManager: fileManager,
            payloadKinds: payloadKinds
        )
    }

    @discardableResult
    internal static func recordRotatingTipExtension(
        previousTip: MLXPersistentPromptCacheTipDescriptor,
        newTip: MLXPersistentPromptCacheTipDescriptor,
        now: Date = Date(),
        fileManager: FileManager = .default,
        compactedPayload: Data = compactedRotatingTipPayload
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        guard isRotatingCacheSignature(newTip.signature),
            previousTip.signature == newTip.signature,
            previousTip.blockSize == newTip.blockSize,
            previousTip.rootURL == newTip.rootURL
        else {
            return nil
        }
        guard let superseded = tipLineageCompactor.recordExtension(
            previousTip: previousTip,
            newTip: newTip
        ) else {
            return nil
        }
        return try rewriteRotatingTipAsCompactedPlaceholder(
            superseded,
            payload: compactedPayload,
            now: now,
            fileManager: fileManager
        )
    }

    @discardableResult
    internal static func storeBlock(
        _ block: MLXPersistentPromptCachePendingBlock,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        fileManager: FileManager = .default
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        try storeBlock(
            blockHash: block.blockHash,
            blockSize: block.blockSize,
            tokenCount: block.tokenCount,
            signature: block.signature,
            payload: block.payload,
            payloadKind: block.payloadKind,
            rootURL: rootURL,
            now: now,
            fileManager: fileManager
        )
    }

    @discardableResult
    internal static func storeBlock(
        blockHash: String,
        blockSize: Int = 256,
        tokenCount: Int,
        signature: PromptCacheSignature,
        payload: Data,
        payloadKind: MLXPersistentPromptCacheBlockPayloadKind = .generic,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        fileManager: FileManager = .default
    ) throws -> MLXPersistentPromptCacheBlockRecord {
        let hash = normalizedHash(blockHash)
        let blockSize = max(1, blockSize)
        let storageHash = storageHash(
            blockHash: hash,
            signature: signature,
            blockSize: blockSize,
            payloadKind: payloadKind
        )
        let directory = blockDirectory(for: hash, rootURL: rootURL)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)

        let dataURL = dataURL(
            forStorageHash: storageHash,
            logicalBlockHash: hash,
            rootURL: rootURL
        )
        try payload.write(to: dataURL, options: .atomic)
        try setAccessDates(now, for: dataURL)
        storeHotPayload(
            payload,
            key: hotPayloadKey(for: dataURL),
            recordPromotion: false
        )

        let record = MLXPersistentPromptCacheBlockRecord(
            blockHash: hash,
            storageHash: storageHash,
            payloadKind: payloadKind,
            blockSize: blockSize,
            tokenCount: tokenCount,
            byteCount: payload.count,
            signature: signature,
            createdAt: now,
            lastAccess: now
        )
        try write(record, rootURL: rootURL)
        MLXGenerationDiagnostics.recordPromptCacheSSDSave()
        return record
    }

    internal static func lookupPrefix(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        fileManager: FileManager = .default,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind] = [.block, .generic]
    ) throws -> MLXPersistentPromptCachePrefixHit? {
        let blockSize = max(1, blockSize)
        let hashes = PromptCacheBlockIndex.prefixBlockHashes(for: tokenIds, blockSize: blockSize)
        guard !hashes.isEmpty else {
            return nil
        }

        var records: [MLXPersistentPromptCacheBlockRecord] = []
        var dataURLs: [URL] = []
        for hash in hashes {
            guard let record = try validRecord(
                blockHash: hash,
                signature: signature,
                blockSize: blockSize,
                rootURL: rootURL,
                fileManager: fileManager,
                payloadKinds: payloadKinds
            ) else {
                break
            }
            let touched = record.accessed(at: now)
            try write(touched, rootURL: rootURL)
            let dataURL = dataURL(for: touched, rootURL: rootURL)
            try setAccessDates(now, for: dataURL)
            records.append(touched)
            dataURLs.append(dataURL)
        }

        guard !records.isEmpty else {
            return nil
        }

        return MLXPersistentPromptCachePrefixHit(
            records: records,
            dataURLs: dataURLs,
            matchedBlockCount: records.count,
            cachedTokenCount: records.reduce(0) { total, record in
                total + record.tokenCount
            },
            nextMissingBlockHash: hashes[safe: records.count]
        )
    }

    internal static func lookupBestPrefixSnapshot(
        tokenIds: [Int],
        signature: PromptCacheSignature,
        blockSize: Int = 256,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        fileManager: FileManager = .default,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind] = [.prefixSnapshot, .generic]
    ) throws -> MLXPersistentPromptCacheSnapshotHit? {
        let blockSize = max(1, blockSize)
        let hashes = PromptCacheBlockIndex.prefixBlockHashes(for: tokenIds, blockSize: blockSize)
        guard !hashes.isEmpty else {
            return nil
        }

        for (offset, hash) in hashes.enumerated().reversed() {
            guard let record = try validRecord(
                blockHash: hash,
                signature: signature,
                blockSize: blockSize,
                rootURL: rootURL,
                fileManager: fileManager,
                payloadKinds: payloadKinds
            ) else {
                continue
            }
            let cachedTokenCount = (offset + 1) * blockSize
            guard record.tokenCount == cachedTokenCount else {
                continue
            }

            let touched = record.accessed(at: now)
            try write(touched, rootURL: rootURL)
            let dataURL = dataURL(for: touched, rootURL: rootURL)
            try setAccessDates(now, for: dataURL)
            return MLXPersistentPromptCacheSnapshotHit(
                record: touched,
                dataURL: dataURL,
                matchedBlockCount: offset + 1,
                cachedTokenCount: cachedTokenCount
            )
        }
        return nil
    }

    internal static func lookup(
        blockHash: String,
        rootURL: URL = rootURL(),
        now: Date = Date(),
        fileManager: FileManager = .default
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        let hash = normalizedHash(blockHash)
        let records = try records(
            logicalBlockHash: hash,
            rootURL: rootURL,
            fileManager: fileManager
        )
        guard let record = records.max(by: { lhs, rhs in
            lhs.lastAccess < rhs.lastAccess
        }) else {
            return nil
        }

        let accessed = record.accessed(at: now)
        try write(accessed, rootURL: rootURL)
        try setAccessDates(now, for: dataURL(for: accessed, rootURL: rootURL))
        return accessed
    }

    internal static func scan(
        rootURL: URL = rootURL(),
        removeStaleFiles: Bool = true,
        fileManager: FileManager = .default
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        guard fileManager.fileExists(atPath: rootURL.path) else {
            return []
        }
        let urls = allFiles(rootURL: rootURL, fileManager: fileManager)
        let metadataURLs = urls.filter { $0.pathExtension == metadataExtension }
        let dataURLs = urls.filter { $0.pathExtension == dataExtension }
        let metadataPaths = Set(metadataURLs.map { url in
            url.deletingPathExtension().standardizedFileURL.path
        })
        var records: [MLXPersistentPromptCacheBlockRecord] = []

        for metadataURL in metadataURLs {
            guard let record = try readRecord(at: metadataURL) else {
                if removeStaleFiles {
                    try? fileManager.removeItem(at: metadataURL)
                }
                continue
            }
            let dataURL = dataURL(for: record, rootURL: rootURL)
            guard fileManager.fileExists(atPath: dataURL.path) else {
                if removeStaleFiles {
                    try? fileManager.removeItem(at: metadataURL)
                }
                continue
            }
            records.append(record)
        }

        if removeStaleFiles {
            for dataURL in dataURLs {
                let key = dataURL.deletingPathExtension().standardizedFileURL.path
                if !metadataPaths.contains(key) {
                    try? fileManager.removeItem(at: dataURL)
                }
            }
        }

        return records.sorted { lhs, rhs in
            if lhs.blockHash == rhs.blockHash {
                return lhs.storageHash < rhs.storageHash
            }
            return lhs.blockHash < rhs.blockHash
        }
    }

    internal static func removeBlock(
        blockHash: String,
        rootURL: URL = rootURL(),
        fileManager: FileManager = .default
    ) {
        let hash = normalizedHash(blockHash)
        if let records = try? records(
            logicalBlockHash: hash,
            rootURL: rootURL,
            fileManager: fileManager
        ) {
            for record in records {
                removeRecord(record, rootURL: rootURL, fileManager: fileManager)
            }
        }
        try? fileManager.removeItem(at: dataURL(for: hash, rootURL: rootURL))
        try? fileManager.removeItem(at: metadataURL(for: hash, rootURL: rootURL))
    }

    internal static func enforceBudget(
        rootURL: URL = rootURL(),
        limitBytes: Int,
        protectedBlockHashes: Set<String> = [],
        protectedStorageHashes: Set<String> = [],
        fileManager: FileManager = .default
    ) throws {
        let protectedBlocks = Set(protectedBlockHashes.map(normalizedHash))
        let protectedStorage = Set(protectedStorageHashes.map(normalizedHash))
        let records = try scan(rootURL: rootURL, removeStaleFiles: true, fileManager: fileManager)
        var totalBytes = records.reduce(0) { $0 + $1.byteCount }
        let removableRecords = records
            .filter { record in
                !isProtected(
                    record,
                    protectedBlocks: protectedBlocks,
                    protectedStorage: protectedStorage
                )
            }
            .sorted { lhs, rhs in
                if lhs.lastAccess == rhs.lastAccess {
                    if lhs.blockHash == rhs.blockHash {
                        return lhs.storageHash < rhs.storageHash
                    }
                    return lhs.blockHash < rhs.blockHash
                }
                return lhs.lastAccess < rhs.lastAccess
            }

        for record in removableRecords where limitBytes <= 0 || totalBytes > limitBytes {
            removeRecord(record, rootURL: rootURL, fileManager: fileManager)
            MLXGenerationDiagnostics.recordPromptCacheEviction()
            totalBytes -= record.byteCount
        }
    }

    private static func isProtected(
        _ record: MLXPersistentPromptCacheBlockRecord,
        protectedBlocks: Set<String>,
        protectedStorage: Set<String>
    ) -> Bool {
        protectedBlocks.contains(record.blockHash) ||
            protectedStorage.contains(record.storageHash)
    }

    @discardableResult
    internal static func invalidateStaleSignatures(
        expectedSignature: PromptCacheSignature,
        rootURL: URL = rootURL(),
        payloadKinds: Set<MLXPersistentPromptCacheBlockPayloadKind>? = nil,
        fileManager: FileManager = .default
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        guard expectedSignature.promptCacheIdentity != nil else {
            recordInvalidation(
                stage: .skippedMissingIdentity,
                candidateCount: 0,
                removedCount: 0,
                payloadKinds: payloadKinds
            )
            return []
        }
        let records = try scan(
            rootURL: rootURL,
            removeStaleFiles: true,
            fileManager: fileManager
        )
        var removedRecords: [MLXPersistentPromptCacheBlockRecord] = []

        for record in records {
            guard expectedSignature.persistentCacheInvalidationScopeMatches(record.signature),
                record.signature != expectedSignature,
                payloadKinds.map({ $0.contains(record.payloadKind) }) ?? true
            else {
                continue
            }
            removeRecord(record, rootURL: rootURL, fileManager: fileManager)
            removedRecords.append(record)
        }
        recordInvalidation(
            stage: .staleSignatureSweep,
            candidateCount: records.count,
            removedCount: removedRecords.count,
            payloadKinds: payloadKinds
        )

        return removedRecords
    }

    private static func recordInvalidation(
        stage: MLXPersistentCacheInvalidationSnapshot.Stage,
        candidateCount: Int,
        removedCount: Int,
        payloadKinds: Set<MLXPersistentPromptCacheBlockPayloadKind>?
    ) {
        MLXGenerationDiagnostics.recordPersistentCacheInvalidation(.init(
            stage: stage,
            candidateCount: candidateCount,
            removedCount: removedCount,
            payloadKinds: payloadKinds?.map(\.rawValue).sorted() ?? []
        ))
    }

    private static func validRecord(
        blockHash: String,
        signature: PromptCacheSignature,
        blockSize: Int,
        rootURL: URL,
        fileManager: FileManager,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind]
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        let hash = normalizedHash(blockHash)
        for payloadKind in payloadKinds {
            let storageHash = storageHash(
                blockHash: hash,
                signature: signature,
                blockSize: blockSize,
                payloadKind: payloadKind
            )
            let dataURL = dataURL(
                forStorageHash: storageHash,
                logicalBlockHash: hash,
                rootURL: rootURL
            )
            let metadataURL = metadataURL(
                forStorageHash: storageHash,
                logicalBlockHash: hash,
                rootURL: rootURL
            )
            if fileManager.fileExists(atPath: dataURL.path),
                fileManager.fileExists(atPath: metadataURL.path),
                let record = try readRecord(at: metadataURL),
                record.signature == signature,
                record.blockSize == max(1, blockSize),
                payloadKinds.contains(record.payloadKind) {
                return record
            }
        }

        if let record = try records(
            logicalBlockHash: hash,
            rootURL: rootURL,
            fileManager: fileManager
        ).first(where: { record in
            record.signature == signature &&
                record.blockSize == max(1, blockSize) &&
                payloadKinds.contains(record.payloadKind)
        }) {
            return record
        }

        return try legacyValidRecord(
            blockHash: hash,
            signature: signature,
            blockSize: blockSize,
            rootURL: rootURL,
            fileManager: fileManager,
            payloadKinds: payloadKinds
        )
    }

    private static func legacyValidRecord(
        blockHash: String,
        signature: PromptCacheSignature,
        blockSize: Int,
        rootURL: URL,
        fileManager: FileManager,
        payloadKinds: [MLXPersistentPromptCacheBlockPayloadKind]
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        let hash = normalizedHash(blockHash)
        let dataURL = dataURL(for: hash, rootURL: rootURL)
        let metadataURL = metadataURL(for: hash, rootURL: rootURL)
        guard fileManager.fileExists(atPath: dataURL.path),
            fileManager.fileExists(atPath: metadataURL.path),
            let record = try readRecord(at: metadataURL),
            record.signature == signature,
            record.blockSize == max(1, blockSize),
            payloadKinds.contains(record.payloadKind)
        else {
            return nil
        }
        return record
    }

    private static func records(
        logicalBlockHash blockHash: String,
        rootURL: URL,
        fileManager: FileManager
    ) throws -> [MLXPersistentPromptCacheBlockRecord] {
        let hash = normalizedHash(blockHash)
        let directory = blockDirectory(for: hash, rootURL: rootURL)
        guard fileManager.fileExists(atPath: directory.path) else {
            return []
        }
        return allFiles(rootURL: directory, fileManager: fileManager)
            .filter { $0.pathExtension == metadataExtension }
            .compactMap { url in
                guard let record = try? readRecord(at: url),
                    record.blockHash == hash,
                    fileManager.fileExists(atPath: dataURL(for: record, rootURL: rootURL).path)
                else {
                    return nil
                }
                return record
            }
    }

    internal static func removeRecord(
        _ record: MLXPersistentPromptCacheBlockRecord,
        rootURL: URL,
        fileManager: FileManager
    ) {
        let dataURL = dataURL(for: record, rootURL: rootURL)
        hotPayloadCache.removeValue(forKey: hotPayloadKey(for: dataURL))
        try? fileManager.removeItem(at: dataURL)
        try? fileManager.removeItem(at: metadataURL(for: record, rootURL: rootURL))
    }

    private static func rewriteRotatingTipAsCompactedPlaceholder(
        _ tip: MLXPersistentPromptCacheTipDescriptor,
        payload: Data,
        now: Date,
        fileManager: FileManager
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        guard let record = try validRecord(
            blockHash: tip.blockHash,
            signature: tip.signature,
            blockSize: tip.blockSize,
            rootURL: tip.rootURL,
            fileManager: fileManager,
            payloadKinds: [tip.payloadKind]
        ), record.storageHash == tip.storageHash else {
            return nil
        }

        let dataURL = dataURL(for: record, rootURL: tip.rootURL)
        try payload.write(to: dataURL, options: .atomic)
        try setAccessDates(now, for: dataURL)
        hotPayloadCache.removeValue(forKey: hotPayloadKey(for: dataURL))
        storeHotPayload(payload, key: hotPayloadKey(for: dataURL), recordPromotion: false)

        let compactedRecord = MLXPersistentPromptCacheBlockRecord(
            blockHash: record.blockHash,
            storageHash: record.storageHash,
            payloadKind: .compactedRotatingTip,
            blockSize: record.blockSize,
            tokenCount: record.tokenCount,
            byteCount: payload.count,
            signature: record.signature,
            createdAt: record.createdAt,
            lastAccess: now
        )
        try write(compactedRecord, rootURL: tip.rootURL)
        return compactedRecord
    }

    private static func isRotatingCacheSignature(_ signature: PromptCacheSignature) -> Bool {
        signature.cacheLayout?.contains { component in
            component.contains("RotatingKVCache")
        } ?? false
    }

    private static func blockDirectory(for blockHash: String, rootURL: URL) -> URL {
        let hash = normalizedHash(blockHash)
        let shard = String(hash.prefix(2))
        return rootURL.appendingPathComponent(shard.isEmpty ? "unknown" : shard, isDirectory: true)
    }

    private static func dataURL(
        forStorageHash storageHash: String,
        logicalBlockHash blockHash: String,
        rootURL: URL
    ) -> URL {
        blockDirectory(for: blockHash, rootURL: rootURL)
            .appendingPathComponent(normalizedHash(storageHash))
            .appendingPathExtension(dataExtension)
    }

    private static func metadataURL(
        forStorageHash storageHash: String,
        logicalBlockHash blockHash: String,
        rootURL: URL
    ) -> URL {
        blockDirectory(for: blockHash, rootURL: rootURL)
            .appendingPathComponent(normalizedHash(storageHash))
            .appendingPathExtension(metadataExtension)
    }

    private static func write(
        _ record: MLXPersistentPromptCacheBlockRecord,
        rootURL: URL
    ) throws {
        let metadataURL = metadataURL(for: record, rootURL: rootURL)
        let data = try JSONEncoder().encode(record)
        try data.write(to: metadataURL, options: .atomic)
        try setAccessDates(record.lastAccess, for: metadataURL)
    }

    private static func readRecord(
        at metadataURL: URL
    ) throws -> MLXPersistentPromptCacheBlockRecord? {
        guard let data = try? Data(contentsOf: metadataURL) else {
            return nil
        }
        return try? JSONDecoder().decode(MLXPersistentPromptCacheBlockRecord.self, from: data)
    }

    private static func allFiles(
        rootURL: URL,
        fileManager: FileManager
    ) -> [URL] {
        guard let urls = try? fileManager.contentsOfDirectory(
            at: rootURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var files = regularFiles(in: urls)
        for directory in shardDirectories(in: urls) {
            guard let shardURLs = try? fileManager.contentsOfDirectory(
                at: directory,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }
            files.append(contentsOf: regularFiles(in: shardURLs))
        }
        return files
    }

    private static func regularFiles(in urls: [URL]) -> [URL] {
        urls.filter { url in
            (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
    }

    private static func shardDirectories(in urls: [URL]) -> [URL] {
        urls.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true &&
                isShardDirectoryName(url.lastPathComponent)
        }
    }

    private static func isShardDirectoryName(_ name: String) -> Bool {
        name == "unknown" || (name.count == 2 && name.allSatisfy { character in
            ("0" ... "9").contains(character) || ("a" ... "f").contains(character)
        })
    }

    private static func setAccessDates(_ date: Date, for url: URL) throws {
        var mutableURL = url
        var values = URLResourceValues()
        values.contentAccessDate = date
        values.contentModificationDate = date
        try mutableURL.setResourceValues(values)
    }

    private static var canonicalEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return encoder
    }

    private static func storeHotPayload(
        _ data: Data,
        key: String,
        recordPromotion: Bool,
        promotionPolicy: MLXPersistentPromptCacheHotPayloadPromotionPolicy = .evictIfNeeded
    ) {
        let evictedCount: Int
        switch promotionPolicy {
        case .evictIfNeeded:
            evictedCount = hotPayloadCache.store(data, forKey: key)

        case .skipIfWouldEvict:
            guard hotPayloadCache.storeIfFits(data, forKey: key) else {
                return
            }
            evictedCount = 0
        }
        if recordPromotion {
            MLXGenerationDiagnostics.recordPromptCacheHotCachePromotion()
        }
        recordHotCacheEvictions(evictedCount)
    }

    private static func recordHotCacheEvictions(_ count: Int) {
        guard count > 0 else {
            return
        }
        for _ in 0 ..< count {
            MLXGenerationDiagnostics.recordPromptCacheHotCacheEviction()
        }
    }

    private static func reservePayload(
        _ record: MLXPersistentPromptCacheBlockRecord,
        dataURL: URL,
        from availableBytes: inout Int
    ) -> Bool {
        let byteCount = payloadByteCount(for: record, dataURL: dataURL)
        guard byteCount <= availableBytes else {
            return false
        }
        availableBytes -= byteCount
        return true
    }

    private static func payloadByteCount(
        for record: MLXPersistentPromptCacheBlockRecord,
        dataURL: URL
    ) -> Int {
        let fileSize = try? dataURL.resourceValues(forKeys: [.fileSizeKey]).fileSize
        return max(record.byteCount, fileSize ?? 0)
    }

    private static func nextMissingBlockHash(
        original hit: MLXPersistentPromptCachePrefixHit,
        selectedCount: Int
    ) -> String? {
        if selectedCount < hit.records.count {
            return hit.records[selectedCount].blockHash
        }
        return hit.nextMissingBlockHash
    }

    private static func hotPayloadKey(for dataURL: URL) -> String {
        dataURL.standardizedFileURL.path
    }
}

private extension SHA256.Digest {
    var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}

private extension Array {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
