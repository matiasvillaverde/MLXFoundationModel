import Foundation

internal enum MLXGenerationAdmissionError: Error, Equatable, Sendable {
    case queueFull(currentDepth: Int, maxDepth: Int)
}

internal actor MLXGenerationAdmissionController {
    internal struct Lease: Equatable, Hashable, Sendable {
        fileprivate let id: Int
    }

    internal struct Snapshot: Equatable, Sendable {
        let activeCount: Int
        let waitingCount: Int
        let maxConcurrentRequests: Int
        let maxQueuedRequests: Int
        let maxBatchSize: Int
        let admissionPaused: Bool
    }

    private enum Waiter {
        case single(id: Int, continuation: CheckedContinuation<Lease, any Error>)
        case batch(
            id: Int,
            requestedCount: Int,
            continuation: CheckedContinuation<[Lease], any Error>
        )

        var id: Int {
            switch self {
            case .single(let id, _), .batch(let id, _, _):
                return id
            }
        }
    }

    private var configuration: MLXGenerationSchedulingConfiguration
    private var admissionPaused = false
    private var activeLeaseIDs: Set<Int> = []
    private var waiters: [Waiter] = []
    private var nextLeaseID = 0

    internal init(
        configuration: MLXGenerationSchedulingConfiguration = .serial
    ) {
        self.configuration = configuration
    }

    internal init(maxConcurrentRequests: Int) {
        self.init(configuration: .init(maxConcurrentRequests: maxConcurrentRequests))
    }

    internal func acquire() async throws -> Lease {
        let id = allocateLeaseID()
        if waiters.isEmpty, canAdmitImmediately {
            return admitLease(id: id)
        }

        guard waiters.count < configuration.maxQueuedRequests else {
            recordAdmission(stage: .queueFull)
            throw MLXGenerationAdmissionError.queueFull(
                currentDepth: waiters.count,
                maxDepth: configuration.maxQueuedRequests
            )
        }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(.single(id: id, continuation: continuation))
                recordAdmission(stage: .queued)
            }
        } onCancel: {
            Task {
                await self.cancelWaiter(id: id)
            }
        }
    }

    internal func acquireBatch(upTo requestedCount: Int) async throws -> [Lease] {
        guard requestedCount > 0 else {
            return []
        }

        if waiters.isEmpty, canAdmitImmediately {
            let id = allocateLeaseID()
            let count = min(requestedCount, configuration.maxBatchSize, availableCapacity)
            return admitBatch(count: count, firstID: id)
        }

        guard waiters.count < configuration.maxQueuedRequests else {
            recordAdmission(stage: .queueFull)
            throw MLXGenerationAdmissionError.queueFull(
                currentDepth: waiters.count,
                maxDepth: configuration.maxQueuedRequests
            )
        }

        let id = allocateLeaseID()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                waiters.append(.batch(
                    id: id,
                    requestedCount: requestedCount,
                    continuation: continuation
                ))
                recordAdmission(stage: .queued)
            }
        } onCancel: {
            Task {
                await self.cancelWaiter(id: id)
            }
        }
    }

    internal func release(_ lease: Lease) {
        guard activeLeaseIDs.remove(lease.id) != nil else {
            return
        }
        recordAdmission(stage: .released)
        admitWaitingRequests()
    }

    internal func updateConfiguration(
        _ configuration: MLXGenerationSchedulingConfiguration
    ) {
        self.configuration = configuration
        recordAdmission(stage: .configured)
        admitWaitingRequests()
    }

    internal func setAdmissionPaused(_ isPaused: Bool) {
        guard admissionPaused != isPaused else {
            return
        }
        admissionPaused = isPaused
        recordAdmission(stage: .pauseUpdated)
        if !isPaused {
            admitWaitingRequests()
        }
    }

    internal func snapshot() -> Snapshot {
        Snapshot(
            activeCount: activeLeaseIDs.count,
            waitingCount: waiters.count,
            maxConcurrentRequests: configuration.maxConcurrentRequests,
            maxQueuedRequests: configuration.maxQueuedRequests,
            maxBatchSize: configuration.maxBatchSize,
            admissionPaused: admissionPaused
        )
    }

    private var canAdmitImmediately: Bool {
        activeLeaseIDs.count < configuration.maxConcurrentRequests
            && !admissionPaused
    }

    private var availableCapacity: Int {
        max(configuration.maxConcurrentRequests - activeLeaseIDs.count, 0)
    }

    private func allocateLeaseID() -> Int {
        defer { nextLeaseID += 1 }
        return nextLeaseID
    }

    private func admitLease(id: Int) -> Lease {
        activeLeaseIDs.insert(id)
        recordAdmission(stage: .admitted, admittedCount: 1)
        return Lease(id: id)
    }

    private func admitBatch(
        count: Int,
        firstID: Int? = nil
    ) -> [Lease] {
        guard count > 0 else {
            return []
        }
        var leases: [Lease] = []
        leases.reserveCapacity(count)
        for index in 0..<count {
            let id: Int
            if index == 0, let firstID {
                id = firstID
            } else {
                id = allocateLeaseID()
            }
            activeLeaseIDs.insert(id)
            leases.append(Lease(id: id))
        }
        recordAdmission(
            stage: count > 1 ? .batchAdmitted : .admitted,
            admittedCount: count
        )
        return leases
    }

    private func cancelWaiter(id: Int) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else {
            return
        }
        let waiter = waiters.remove(at: index)
        switch waiter {
        case .single(_, let continuation):
            continuation.resume(throwing: CancellationError())

        case .batch(_, _, let continuation):
            continuation.resume(throwing: CancellationError())
        }
    }

    private func admitWaitingRequests() {
        while canAdmitImmediately, !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            switch waiter {
            case .single(let id, let continuation):
                let lease = admitLease(id: id)
                continuation.resume(returning: lease)

            case .batch(let id, let requestedCount, let continuation):
                let count = min(requestedCount, configuration.maxBatchSize, availableCapacity)
                let leases = admitBatch(count: count, firstID: id)
                continuation.resume(returning: leases)
            }
        }
    }

    private func recordAdmission(
        stage: MLXGenerationAdmissionSnapshot.Stage,
        admittedCount: Int = 0
    ) {
        MLXGenerationDiagnostics.recordAdmission(MLXGenerationAdmissionSnapshot(
            stage: stage,
            activeCount: activeLeaseIDs.count,
            waitingCount: waiters.count,
            maxConcurrentRequests: configuration.maxConcurrentRequests,
            maxQueuedRequests: configuration.maxQueuedRequests,
            maxBatchSize: configuration.maxBatchSize,
            admittedCount: admittedCount,
            admissionPaused: admissionPaused
        ))
    }
}

internal final class MLXGenerationCancellationState: @unchecked Sendable {
    private let lock = NSLock()
    private var isActive = false

    internal init() {
        lock.name = "org.mlxfoundationmodel.generation-cancellation"
    }

    internal func activate() {
        setActive(true)
    }

    internal func deactivate() {
        setActive(false)
    }

    internal func shouldSignalStop() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return isActive
    }

    private func setActive(_ newValue: Bool) {
        lock.lock()
        defer { lock.unlock() }
        isActive = newValue
    }
}
