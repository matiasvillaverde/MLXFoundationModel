internal struct MLXGenerationBatchRowID: Hashable, Comparable, Sendable,
    CustomStringConvertible, ExpressibleByIntegerLiteral
{
    internal let rawValue: Int

    internal init(_ rawValue: Int) {
        self.rawValue = rawValue
    }

    internal init(integerLiteral value: Int) {
        self.rawValue = value
    }

    internal static func < (
        lhs: MLXGenerationBatchRowID,
        rhs: MLXGenerationBatchRowID
    ) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    internal var description: String {
        String(rawValue)
    }
}

internal struct MLXGenerationBatchRow<Payload: Sendable>: Sendable {
    internal let id: MLXGenerationBatchRowID
    internal var payload: Payload

    internal init(id: MLXGenerationBatchRowID, payload: Payload) {
        self.id = id
        self.payload = payload
    }
}

internal enum MLXGenerationBatchRowTableError: Error, Equatable {
    case duplicateRowID(MLXGenerationBatchRowID)
    case missingRowID(MLXGenerationBatchRowID)
    case payloadCountMismatch(expected: Int, actual: Int)
}

internal struct MLXGenerationBatchRowTable<Payload: Sendable>: Sendable {
    private var rows: [MLXGenerationBatchRow<Payload>] = []
    private var indexByID: [MLXGenerationBatchRowID: Int] = [:]

    internal init() {}

    internal var isEmpty: Bool {
        rows.isEmpty
    }

    internal var count: Int {
        rows.count
    }

    internal var orderedRows: [MLXGenerationBatchRow<Payload>] {
        rows
    }

    internal var orderedIDs: [MLXGenerationBatchRowID] {
        rows.map(\.id)
    }

    internal var orderedPayloads: [Payload] {
        rows.map(\.payload)
    }

    internal subscript(id: MLXGenerationBatchRowID) -> Payload? {
        guard let index = indexByID[id] else {
            return nil
        }
        return rows[index].payload
    }

    internal mutating func append(
        id: MLXGenerationBatchRowID,
        payload: Payload
    ) throws {
        try append(.init(id: id, payload: payload))
    }

    internal mutating func append(
        _ row: MLXGenerationBatchRow<Payload>
    ) throws {
        guard indexByID[row.id] == nil else {
            throw MLXGenerationBatchRowTableError.duplicateRowID(row.id)
        }

        indexByID[row.id] = rows.count
        rows.append(row)
        record(stage: .appended, affectedIDs: [row.id])
    }

    internal mutating func updatePayload(
        for id: MLXGenerationBatchRowID,
        _ update: (inout Payload) throws -> Void
    ) throws {
        guard let index = indexByID[id] else {
            throw MLXGenerationBatchRowTableError.missingRowID(id)
        }

        try update(&rows[index].payload)
        record(stage: .updated, affectedIDs: [id])
    }

    @discardableResult
    internal mutating func remove(id: MLXGenerationBatchRowID) -> MLXGenerationBatchRow<Payload>? {
        guard let index = indexByID[id] else {
            return nil
        }

        let row = rows.remove(at: index)
        rebuildIndex()
        record(stage: .removed, affectedIDs: [row.id])
        return row
    }

    @discardableResult
    internal mutating func remove(
        ids: Set<MLXGenerationBatchRowID>
    ) -> [MLXGenerationBatchRow<Payload>] {
        guard !ids.isEmpty else {
            return []
        }

        var keptRows: [MLXGenerationBatchRow<Payload>] = []
        var removedRows: [MLXGenerationBatchRow<Payload>] = []
        keptRows.reserveCapacity(rows.count)

        for row in rows {
            if ids.contains(row.id) {
                removedRows.append(row)
            } else {
                keptRows.append(row)
            }
        }

        guard !removedRows.isEmpty else {
            return []
        }

        rows = keptRows
        rebuildIndex()
        record(stage: .removed, affectedIDs: removedRows.map(\.id))
        return removedRows
    }

    internal mutating func keep(
        ids idsToKeep: [MLXGenerationBatchRowID]
    ) throws {
        var seenIDs: Set<MLXGenerationBatchRowID> = []
        var keptRows: [MLXGenerationBatchRow<Payload>] = []
        keptRows.reserveCapacity(idsToKeep.count)

        for id in idsToKeep {
            guard seenIDs.insert(id).inserted else {
                throw MLXGenerationBatchRowTableError.duplicateRowID(id)
            }
            guard let index = indexByID[id] else {
                throw MLXGenerationBatchRowTableError.missingRowID(id)
            }
            keptRows.append(rows[index])
        }

        let removedIDs = Set(indexByID.keys).subtracting(seenIDs)
        rows = keptRows
        rebuildIndex()
        record(stage: .kept, affectedIDs: Array(removedIDs).sorted())
    }

    internal mutating func replaceOrderedPayloads(_ payloads: [Payload]) throws {
        guard payloads.count == rows.count else {
            throw MLXGenerationBatchRowTableError.payloadCountMismatch(
                expected: rows.count,
                actual: payloads.count
            )
        }
        guard !payloads.isEmpty else {
            return
        }

        for index in rows.indices {
            rows[index].payload = payloads[index]
        }
        record(stage: .updated, affectedIDs: rows.map(\.id))
    }

    private mutating func rebuildIndex() {
        indexByID.removeAll(keepingCapacity: true)
        for (index, row) in rows.enumerated() {
            indexByID[row.id] = index
        }
    }

    private func record(
        stage: MLXGenerationBatchRowsSnapshot.Stage,
        affectedIDs: [MLXGenerationBatchRowID]
    ) {
        MLXGenerationDiagnostics.recordBatchRows(.init(
            stage: stage,
            rowCount: rows.count,
            rowIDs: rows.map(\.id.rawValue),
            affectedRowIDs: affectedIDs.map(\.rawValue)
        ))
    }
}
