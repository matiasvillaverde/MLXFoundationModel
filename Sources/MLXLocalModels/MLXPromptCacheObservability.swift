import Foundation

internal struct MLXPromptCacheObservabilityCounters: Equatable, Sendable {
    var prefixHits: Int = 0
    var prefixMisses: Int = 0
    var prefixTokensMatched: Int = 0
    var prefixTokensRequested: Int = 0
    var prefixTokensSaved: Int = 0
    var evictions: Int = 0
    var ssdHotHits: Int = 0
    var ssdDiskLoads: Int = 0
    var ssdSaves: Int = 0
    var hotCacheEvictions: Int = 0
    var hotCachePromotions: Int = 0

    var prefixHitRate: Double {
        Self.roundedRatio(prefixHits, prefixHits + prefixMisses)
    }

    var prefixMatchEfficiency: Double {
        Self.roundedRatio(prefixTokensMatched, prefixTokensRequested)
    }

    var ssdHotRate: Double {
        Self.roundedRatio(ssdHotHits, ssdHotHits + ssdDiskLoads)
    }

    private static func roundedRatio(_ numerator: Int, _ denominator: Int) -> Double {
        guard denominator > 0 else {
            return 0
        }
        return (Double(numerator) / Double(denominator) * 10_000).rounded() / 10_000
    }
}

internal struct MLXPromptCacheRateWindow: Equatable, Sendable {
    let label: String
    let elapsedSeconds: Double
    let prefixHits: Int
    let prefixMisses: Int
    let prefixHitRate: Double
    let prefixTokensMatched: Int
    let prefixTokensRequested: Int
    let prefixMatchEfficiency: Double
    let evictions: Int
    let evictionRatePerMinute: Double
    let ssdHotHits: Int
    let ssdDiskLoads: Int
    let ssdHotRate: Double
}

internal struct MLXPromptCacheObservabilitySnapshot: Equatable, Sendable {
    let counters: MLXPromptCacheObservabilityCounters
    let windows: [String: MLXPromptCacheRateWindow]
}

internal final class MLXPromptCacheObservabilityTracker: @unchecked Sendable {
    private struct Observation {
        let timestamp: Double
        let counters: MLXPromptCacheObservabilityCounters
    }

    private let lock = NSLock()
    private let maxSnapshots: Int
    private let minSnapshotInterval: Double
    private var counters = MLXPromptCacheObservabilityCounters()
    private var observations: [Observation] = []

    internal init(maxSnapshots: Int = 90, minSnapshotInterval: Double = 10) {
        self.maxSnapshots = max(2, maxSnapshots)
        self.minSnapshotInterval = max(0, minSnapshotInterval)
        lock.name = "org.mlxfoundationmodel.prompt-cache-observability"
    }

    @discardableResult
    internal func recordPlan(
        promptTokenCount: Int,
        reusedTokenCount: Int,
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        updatePrefixCounters(
            promptTokenCount: promptTokenCount,
            reusedTokenCount: reusedTokenCount
        )
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordEviction(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.evictions += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordSSDHotHit(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.ssdHotHits += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordSSDDiskLoad(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.ssdDiskLoads += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordSSDSave(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.ssdSaves += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordHotCacheEviction(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.hotCacheEvictions += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    @discardableResult
    internal func recordHotCachePromotion(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        counters.hotCachePromotions += 1
        appendObservationIfNeeded(timestamp: timestamp)
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    internal func snapshot(
        timestamp: Double = ProcessInfo.processInfo.systemUptime
    ) -> MLXPromptCacheObservabilitySnapshot {
        lock.lock()
        let snapshot = makeSnapshot(timestamp: timestamp)
        lock.unlock()
        return snapshot
    }

    internal func reset() {
        lock.lock()
        counters = MLXPromptCacheObservabilityCounters()
        observations.removeAll(keepingCapacity: true)
        lock.unlock()
    }

    private func updatePrefixCounters(
        promptTokenCount: Int,
        reusedTokenCount: Int
    ) {
        let requestedTokenCount = max(promptTokenCount - 1, 0)
        guard requestedTokenCount > 0 else {
            return
        }

        let matchedTokenCount = min(max(reusedTokenCount, 0), requestedTokenCount)
        if matchedTokenCount > 0 {
            counters.prefixHits += 1
        } else {
            counters.prefixMisses += 1
        }
        counters.prefixTokensMatched += matchedTokenCount
        counters.prefixTokensRequested += requestedTokenCount
        counters.prefixTokensSaved += matchedTokenCount
    }

    private func appendObservationIfNeeded(timestamp: Double) {
        if let last = observations.last,
            timestamp - last.timestamp < minSnapshotInterval {
            return
        }

        observations.append(Observation(timestamp: timestamp, counters: counters))
        if observations.count > maxSnapshots {
            observations.removeFirst(observations.count - maxSnapshots)
        }
    }

    private func makeSnapshot(timestamp: Double) -> MLXPromptCacheObservabilitySnapshot {
        MLXPromptCacheObservabilitySnapshot(
            counters: counters,
            windows: rateWindows(timestamp: timestamp)
        )
    }

    private func rateWindows(timestamp: Double) -> [String: MLXPromptCacheRateWindow] {
        guard !observations.isEmpty else {
            return [:]
        }

        var windows: [String: MLXPromptCacheRateWindow] = [:]
        for windowSeconds in [60, 300, 900] {
            let label = Self.windowLabel(seconds: windowSeconds)
            let baseline = baselineObservation(
                timestamp: timestamp,
                windowSeconds: Double(windowSeconds)
            )
            let elapsed = max(timestamp - baseline.timestamp, 0)
            guard elapsed >= 1 else {
                continue
            }
            windows[label] = Self.window(
                label: label,
                baseline: baseline.counters,
                current: counters,
                elapsed: elapsed
            )
        }
        return windows
    }

    private func baselineObservation(
        timestamp: Double,
        windowSeconds: Double
    ) -> Observation {
        observations.first { observation in
            timestamp - observation.timestamp <= windowSeconds
        } ?? observations[0]
    }

    private static func window(
        label: String,
        baseline: MLXPromptCacheObservabilityCounters,
        current: MLXPromptCacheObservabilityCounters,
        elapsed: Double
    ) -> MLXPromptCacheRateWindow {
        let prefixHits = delta(current.prefixHits, baseline.prefixHits)
        let prefixMisses = delta(current.prefixMisses, baseline.prefixMisses)
        let prefixTokensMatched = delta(
            current.prefixTokensMatched,
            baseline.prefixTokensMatched
        )
        let prefixTokensRequested = delta(
            current.prefixTokensRequested,
            baseline.prefixTokensRequested
        )
        let evictions = delta(current.evictions, baseline.evictions)
        let ssdHotHits = delta(current.ssdHotHits, baseline.ssdHotHits)
        let ssdDiskLoads = delta(current.ssdDiskLoads, baseline.ssdDiskLoads)

        return MLXPromptCacheRateWindow(
            label: label,
            elapsedSeconds: elapsed,
            prefixHits: prefixHits,
            prefixMisses: prefixMisses,
            prefixHitRate: roundedRatio(prefixHits, prefixHits + prefixMisses),
            prefixTokensMatched: prefixTokensMatched,
            prefixTokensRequested: prefixTokensRequested,
            prefixMatchEfficiency: roundedRatio(prefixTokensMatched, prefixTokensRequested),
            evictions: evictions,
            evictionRatePerMinute: rounded(Double(evictions) / elapsed * 60),
            ssdHotHits: ssdHotHits,
            ssdDiskLoads: ssdDiskLoads,
            ssdHotRate: roundedRatio(ssdHotHits, ssdHotHits + ssdDiskLoads)
        )
    }

    private static func windowLabel(seconds: Int) -> String {
        seconds < 60 ? "\(seconds)s" : "\(seconds / 60)m"
    }

    private static func delta(_ current: Int, _ baseline: Int) -> Int {
        max(0, current - baseline)
    }

    private static func roundedRatio(_ numerator: Int, _ denominator: Int) -> Double {
        guard denominator > 0 else {
            return 0
        }
        return rounded(Double(numerator) / Double(denominator))
    }

    private static func rounded(_ value: Double) -> Double {
        (value * 10_000).rounded() / 10_000
    }
}
