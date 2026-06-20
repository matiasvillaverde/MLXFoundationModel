import Foundation
@testable import MLXFoundationModel
import MLXLocalModels
import Testing

@Suite("MLX model pool memory admission")
struct MLXModelPoolMemoryTests {
    @Test("evicts least recently used residents to satisfy byte budget")
    func evictsLeastRecentlyUsedResidentsToSatisfyByteBudget() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 3, maxResidentMemoryBytes: 90),
            sessionFactory: store.makeSession
        )

        try await Self.register(["a": 40, "b": 30, "c": 50], in: pool, root: root)
        _ = try await pool.preload(id: "a", now: Self.time(0))
        _ = try await pool.preload(id: "b", now: Self.time(1))
        _ = try await pool.preload(id: "a", now: Self.time(2))
        _ = try await pool.preload(id: "c", now: Self.time(3))

        let snapshot = await pool.snapshot()
        let sessions = store.snapshot()

        #expect(snapshot.residentModelIDs == ["a", "c"])
        #expect(snapshot.residentMemoryBytes == 90)
        #expect(snapshot.residentMemoryBytesByModelID == ["a": 40, "c": 50])
        #expect(await sessions[1].unloadCount == 1)
    }

    @Test("pinned residents block byte-budget eviction")
    func pinnedResidentsBlockByteBudgetEviction() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 2, maxResidentMemoryBytes: 100),
            sessionFactory: store.makeSession
        )

        try await pool.register(Self.model("pinned", bytes: 80, root: root, pinned: true))
        try await pool.register(Self.model("other", bytes: 40, root: root))
        _ = try await pool.preload(id: "pinned", now: Self.time(0))

        await #expect(throws: Self.memoryError(requested: 40, limit: 100, resident: 80)) {
            _ = try await pool.preload(id: "other", now: Self.time(1))
        }

        let snapshot = await pool.snapshot()
        #expect(snapshot.residentModelIDs == ["pinned"])
        #expect(store.snapshot().count == 1)
    }

    @Test("leased residents block byte-budget eviction until released")
    func leasedResidentsBlockByteBudgetEvictionUntilReleased() async throws {
        let store = MLXModelPoolRecordingSessionStore()
        let root = try Self.makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let pool = MLXModelPool(
            configuration: .init(maxResidentModels: 2, maxResidentMemoryBytes: 100),
            sessionFactory: store.makeSession
        )

        try await Self.register(["active": 80, "other": 40], in: pool, root: root)
        _ = try await pool.withSession(id: "active", now: Self.time(0)) { _ in
            await #expect(throws: Self.memoryError(requested: 40, limit: 100, resident: 80)) {
                _ = try await pool.preload(id: "other", now: Self.time(1))
            }
        }

        let snapshot = await pool.snapshot()
        #expect(snapshot.residentModelIDs == ["active"])
        #expect(snapshot.leasedResidentModelIDs.isEmpty)
    }

    private static func register(
        _ models: [String: Int],
        in pool: MLXModelPool,
        root: URL
    ) async throws {
        for (id, bytes) in models {
            try await pool.register(Self.model(id, bytes: bytes, root: root))
        }
    }

    private static func model(
        _ id: String,
        bytes: Int,
        root: URL,
        pinned: Bool = false
    ) throws -> MLXLanguageModel {
        let directory = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(repeating: UInt8(bytes % 255), count: bytes)
            .write(to: directory.appendingPathComponent("model.safetensors"))
        return MLXLanguageModel(
            model: MLXModel(id: id, location: directory),
            runtime: ModelRuntimePreferences(isPinned: pinned)
        )
    }

    private static func memoryError(
        requested: Int,
        limit: Int,
        resident: Int
    ) -> MLXModelPoolError {
        .residentMemoryCapacityExhausted(
            requestedBytes: requested,
            maxResidentMemoryBytes: limit,
            residentBytes: resident
        )
    }

    private static func makeTemporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private static func time(_ seconds: TimeInterval) -> Date {
        Date(timeIntervalSince1970: seconds)
    }
}
