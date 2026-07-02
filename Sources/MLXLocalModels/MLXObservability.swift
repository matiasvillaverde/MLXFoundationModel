import Foundation
import OSLog

/// Stable observability buckets used for logs, signposts, counters, and snapshots.
public enum MLXObservabilityCategory: String, Sendable, Codable, CaseIterable {
    case generation
    case modelLoad
    case modelPool
    case promptCache
    case memoryGuard
    case provider
    case toolParser
    case grammar
    case streaming
    case admission
    case runtime
}

/// Central severity levels for structured logs and sink events.
public enum MLXObservabilitySeverity: String, Sendable, Codable, CaseIterable, Comparable {
    case debug
    case info
    case notice
    case warning
    case error
    case fault

    public static func < (
        lhs: MLXObservabilitySeverity,
        rhs: MLXObservabilitySeverity
    ) -> Bool {
        lhs.rank < rhs.rank
    }

    private var rank: Int {
        switch self {
        case .debug:
            return 0
        case .info:
            return 1
        case .notice:
            return 2
        case .warning:
            return 3
        case .error:
            return 4
        case .fault:
            return 5
        }
    }
}

/// Structured event kinds recorded by ``MLXObservability``.
public enum MLXObservabilityEventKind: String, Sendable, Codable, CaseIterable {
    case log
    case span
    case counter
    case gauge
    case histogram
    case requestSummary
    case diagnostic
}

/// Coarse-grained trace spans emitted as OS signposts and structured events.
public enum MLXTraceSpan: String, Sendable, Codable, CaseIterable {
    case modelLoad = "model.load"
    case modelUnload = "model.unload"
    case generation = "generation"
    case requestRender = "request.render"
    case tokenize = "tokenize"
    case admissionWait = "admission.wait"
    case promptCacheLookup = "prompt_cache.lookup"
    case promptCacheRestore = "prompt_cache.restore"
    case memoryGuardCheck = "memory_guard.check"
    case prefill = "prefill"
    case firstToken = "first_token"
    case decode = "decode"
    case grammarMask = "grammar.mask"
    case toolParse = "tool.parse"
    case streamTranslate = "stream.translate"
    case promptCacheSave = "prompt_cache.save"
    case continuousBatch = "continuous_batch"

    public var category: MLXObservabilityCategory {
        switch self {
        case .modelLoad, .modelUnload:
            return .modelLoad
        case .requestRender, .streamTranslate:
            return .provider
        case .promptCacheLookup, .promptCacheRestore, .promptCacheSave:
            return .promptCache
        case .memoryGuardCheck:
            return .memoryGuard
        case .grammarMask:
            return .grammar
        case .toolParse:
            return .toolParser
        case .admissionWait:
            return .admission
        case .generation, .tokenize, .prefill, .firstToken, .decode, .continuousBatch:
            return .generation
        }
    }

    internal var signpostName: StaticString {
        switch self {
        case .modelLoad:
            return "model.load"
        case .modelUnload:
            return "model.unload"
        case .generation:
            return "generation"
        case .requestRender:
            return "request.render"
        case .tokenize:
            return "tokenize"
        case .admissionWait:
            return "admission.wait"
        case .promptCacheLookup:
            return "prompt_cache.lookup"
        case .promptCacheRestore:
            return "prompt_cache.restore"
        case .memoryGuardCheck:
            return "memory_guard.check"
        case .prefill:
            return "prefill"
        case .firstToken:
            return "first_token"
        case .decode:
            return "decode"
        case .grammarMask:
            return "grammar.mask"
        case .toolParse:
            return "tool.parse"
        case .streamTranslate:
            return "stream.translate"
        case .promptCacheSave:
            return "prompt_cache.save"
        case .continuousBatch:
            return "continuous_batch"
        }
    }
}

/// One structured observability event suitable for custom sinks and snapshots.
public struct MLXObservabilityEvent: Sendable, Codable, Equatable {
    public let name: String
    public let kind: MLXObservabilityEventKind
    public let category: MLXObservabilityCategory
    public let severity: MLXObservabilitySeverity
    public let uptimeSeconds: Double
    public let durationSeconds: Double?
    public let attributes: [String: String]
    public let measurements: [String: Double]

    public init(
        name: String,
        kind: MLXObservabilityEventKind,
        category: MLXObservabilityCategory,
        severity: MLXObservabilitySeverity = .info,
        uptimeSeconds: Double = ProcessInfo.processInfo.systemUptime,
        durationSeconds: Double? = nil,
        attributes: [String: String] = [:],
        measurements: [String: Double] = [:]
    ) {
        self.name = name
        self.kind = kind
        self.category = category
        self.severity = severity
        self.uptimeSeconds = uptimeSeconds
        self.durationSeconds = durationSeconds
        self.attributes = attributes
        self.measurements = measurements
    }
}

/// Aggregated metric summary for histograms retained in memory.
public struct MLXMetricSummary: Sendable, Codable, Equatable {
    public let count: Int
    public let sum: Double
    public let min: Double
    public let max: Double
    public let average: Double

    public init(count: Int, sum: Double, min: Double, max: Double, average: Double) {
        self.count = count
        self.sum = sum
        self.min = min
        self.max = max
        self.average = average
    }
}

/// Redacted per-request generation summary.
public struct MLXRequestSummary: Sendable, Codable, Equatable {
    public let requestID: UUID
    public let modelName: String?
    public let strategy: String?
    public let promptTokens: Int
    public let generatedTokens: Int
    public let totalTokens: Int
    public let cachedPromptTokens: Int?
    public let totalDurationSeconds: Double
    public let timeToFirstTokenSeconds: Double?
    public let promptProcessingSeconds: Double?
    public let promptTokensPerSecond: Double?
    public let generationTokensPerSecond: Double?
    public let totalTokensPerSecond: Double?
    public let kvCacheBytes: Int64?
    public let kvCacheEntries: Int?
    public let stopReason: String
    public let temperature: Double
    public let topP: Double
    public let topK: Int?
    public let grammarKind: String?

    public init(
        requestID: UUID = UUID(),
        modelName: String? = nil,
        strategy: String? = nil,
        promptTokens: Int,
        generatedTokens: Int,
        totalTokens: Int,
        cachedPromptTokens: Int? = nil,
        totalDurationSeconds: Double,
        timeToFirstTokenSeconds: Double? = nil,
        promptProcessingSeconds: Double? = nil,
        promptTokensPerSecond: Double? = nil,
        generationTokensPerSecond: Double? = nil,
        totalTokensPerSecond: Double? = nil,
        kvCacheBytes: Int64? = nil,
        kvCacheEntries: Int? = nil,
        stopReason: String,
        temperature: Double,
        topP: Double,
        topK: Int? = nil,
        grammarKind: String? = nil
    ) {
        self.requestID = requestID
        self.modelName = modelName
        self.strategy = strategy
        self.promptTokens = promptTokens
        self.generatedTokens = generatedTokens
        self.totalTokens = totalTokens
        self.cachedPromptTokens = cachedPromptTokens
        self.totalDurationSeconds = totalDurationSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.promptProcessingSeconds = promptProcessingSeconds
        self.promptTokensPerSecond = promptTokensPerSecond
        self.generationTokensPerSecond = generationTokensPerSecond
        self.totalTokensPerSecond = totalTokensPerSecond
        self.kvCacheBytes = kvCacheBytes
        self.kvCacheEntries = kvCacheEntries
        self.stopReason = stopReason
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.grammarKind = grammarKind
    }
}

/// Current in-memory counters, gauges, histograms, and retained recent events.
public struct MLXObservabilitySnapshot: Sendable, Codable, Equatable {
    public let counters: [String: Double]
    public let gauges: [String: Double]
    public let histograms: [String: MLXMetricSummary]
    public let recentEvents: [MLXObservabilityEvent]
    public let recentRequests: [MLXRequestSummary]

    public init(
        counters: [String: Double],
        gauges: [String: Double],
        histograms: [String: MLXMetricSummary],
        recentEvents: [MLXObservabilityEvent],
        recentRequests: [MLXRequestSummary]
    ) {
        self.counters = counters
        self.gauges = gauges
        self.histograms = histograms
        self.recentEvents = recentEvents
        self.recentRequests = recentRequests
    }
}

/// Runtime configuration for the central observability registry.
public struct MLXObservabilityConfiguration: Sendable, Codable, Equatable {
    public let isEnabled: Bool
    public let osLogEnabled: Bool
    public let signpostsEnabled: Bool
    public let minimumLogSeverity: MLXObservabilitySeverity
    public let subsystem: String
    public let keptRecentEventCount: Int
    public let keptRecentRequestCount: Int

    public init(
        isEnabled: Bool = true,
        osLogEnabled: Bool = true,
        signpostsEnabled: Bool = true,
        minimumLogSeverity: MLXObservabilitySeverity = .info,
        subsystem: String = MLXObservability.defaultSubsystem,
        keptRecentEventCount: Int = 256,
        keptRecentRequestCount: Int = 64
    ) {
        self.isEnabled = isEnabled
        self.osLogEnabled = osLogEnabled
        self.signpostsEnabled = signpostsEnabled
        self.minimumLogSeverity = minimumLogSeverity
        self.subsystem = subsystem
        self.keptRecentEventCount = max(0, keptRecentEventCount)
        self.keptRecentRequestCount = max(0, keptRecentRequestCount)
    }

    public static let `default` = MLXObservabilityConfiguration()
}

/// Optional sink for exporting observability events to files, telemetry, or tests.
public protocol MLXObservabilitySink: Sendable {
    func record(_ event: MLXObservabilityEvent)
    func recordRequest(_ summary: MLXRequestSummary)
}

public extension MLXObservabilitySink {
    func recordRequest(_ summary: MLXRequestSummary) {
        _ = summary
    }
}

/// Active coarse-grained trace interval.
public struct MLXObservabilitySpan {
    private let traceSpan: MLXTraceSpan
    private let attributes: [String: String]
    private let clock: ContinuousClock
    private let start: ContinuousClock.Instant
    private let signpost: MLXActiveSignpost?

    fileprivate init(
        traceSpan: MLXTraceSpan,
        attributes: [String: String],
        signpost: MLXActiveSignpost?
    ) {
        self.traceSpan = traceSpan
        self.attributes = attributes
        self.clock = ContinuousClock()
        self.start = clock.now
        self.signpost = signpost
    }

    public func end() {
        let duration = start.duration(to: clock.now).mlxSeconds
        if let signpost {
            signpost.end(traceSpan)
        }
        MLXObservability.record(MLXObservabilityEvent(
            name: traceSpan.rawValue,
            kind: .span,
            category: traceSpan.category,
            severity: .debug,
            durationSeconds: duration,
            attributes: attributes,
            measurements: ["duration_seconds": duration]
        ))
        MLXObservability.recordHistogram(
            "span.\(traceSpan.rawValue).duration_seconds",
            value: duration,
            category: traceSpan.category,
            attributes: attributes
        )
    }
}

private struct MLXActiveSignpost {
    let signposter: OSSignposter
    let state: OSSignpostIntervalState

    func end(_ span: MLXTraceSpan) {
        signposter.endInterval(span.signpostName, state)
    }
}

/// Central in-process observability registry for MLX local inference.
public enum MLXObservability {
    public static let defaultSubsystem = "org.mlxfoundationmodel"

    private static let store = MLXObservabilityStore()

    public static func configure(
        _ configuration: MLXObservabilityConfiguration = .default,
        sink: (any MLXObservabilitySink)? = nil
    ) {
        store.configure(configuration, sink: sink)
    }

    public static func reset(
        keepingConfiguration: Bool = false
    ) {
        store.reset(keepingConfiguration: keepingConfiguration)
    }

    public static func snapshot() -> MLXObservabilitySnapshot {
        store.snapshot()
    }

    public static func logger(
        for category: MLXObservabilityCategory
    ) -> Logger {
        Logger(subsystem: store.configuration.subsystem, category: category.rawValue)
    }

    public static func record(
        _ event: MLXObservabilityEvent
    ) {
        store.record(event)
    }

    public static func log(
        _ name: String,
        category: MLXObservabilityCategory,
        severity: MLXObservabilitySeverity = .info,
        attributes: [String: String] = [:],
        measurements: [String: Double] = [:]
    ) {
        record(MLXObservabilityEvent(
            name: name,
            kind: .log,
            category: category,
            severity: severity,
            attributes: attributes,
            measurements: measurements
        ))
    }

    public static func incrementCounter(
        _ name: String,
        by amount: Double = 1,
        category: MLXObservabilityCategory,
        attributes: [String: String] = [:]
    ) {
        store.incrementCounter(name, by: amount)
        record(MLXObservabilityEvent(
            name: name,
            kind: .counter,
            category: category,
            severity: .debug,
            attributes: attributes,
            measurements: ["value": amount]
        ))
    }

    public static func setGauge(
        _ name: String,
        value: Double,
        category: MLXObservabilityCategory,
        attributes: [String: String] = [:]
    ) {
        store.setGauge(name, value: value)
        record(MLXObservabilityEvent(
            name: name,
            kind: .gauge,
            category: category,
            severity: .debug,
            attributes: attributes,
            measurements: ["value": value]
        ))
    }

    public static func recordHistogram(
        _ name: String,
        value: Double,
        category: MLXObservabilityCategory,
        attributes: [String: String] = [:]
    ) {
        store.recordHistogram(name, value: value)
        record(MLXObservabilityEvent(
            name: name,
            kind: .histogram,
            category: category,
            severity: .debug,
            attributes: attributes,
            measurements: ["value": value]
        ))
    }

    public static func recordRequestSummary(
        _ summary: MLXRequestSummary
    ) {
        store.recordRequestSummary(summary)
    }

    public static func startSpan(
        _ span: MLXTraceSpan,
        attributes: [String: String] = [:]
    ) -> MLXObservabilitySpan {
        let configuration = store.configuration
        return MLXObservabilitySpan(
            traceSpan: span,
            attributes: attributes,
            signpost: makeSignpost(span, configuration: configuration)
        )
    }

    public static func withSpan<T>(
        _ span: MLXTraceSpan,
        attributes: [String: String] = [:],
        operation: () throws -> T
    ) rethrows -> T {
        let activeSpan = startSpan(span, attributes: attributes)
        defer {
            activeSpan.end()
        }
        return try operation()
    }

    public static func withAsyncSpan<T>(
        _ span: MLXTraceSpan,
        attributes: [String: String] = [:],
        operation: () async throws -> T
    ) async rethrows -> T {
        let activeSpan = startSpan(span, attributes: attributes)
        defer {
            activeSpan.end()
        }
        return try await operation()
    }

    private static func makeSignpost(
        _ span: MLXTraceSpan,
        configuration: MLXObservabilityConfiguration
    ) -> MLXActiveSignpost? {
        guard configuration.signpostsEnabled, configuration.isEnabled else {
            return nil
        }
        let signposter = OSSignposter(logger: logger(for: span.category))
        let id = signposter.makeSignpostID()
        return MLXActiveSignpost(
            signposter: signposter,
            state: signposter.beginInterval(span.signpostName, id: id)
        )
    }
}

private struct MLXHistogramAccumulator: Sendable, Equatable {
    var count: Int = 0
    var sum: Double = 0
    var min: Double = .infinity
    var max: Double = -.infinity

    mutating func record(_ value: Double) {
        count += 1
        sum += value
        min = Swift.min(min, value)
        max = Swift.max(max, value)
    }

    var summary: MLXMetricSummary {
        MLXMetricSummary(
            count: count,
            sum: sum,
            min: count > 0 ? min : 0,
            max: count > 0 ? max : 0,
            average: count > 0 ? sum / Double(count) : 0
        )
    }
}

private final class MLXObservabilityStore: @unchecked Sendable {
    private let lock = NSLock()
    private var currentConfiguration = MLXObservabilityConfiguration.default
    private var currentSink: (any MLXObservabilitySink)?
    private var counters: [String: Double] = [:]
    private var gauges: [String: Double] = [:]
    private var histograms: [String: MLXHistogramAccumulator] = [:]
    private var recentEvents: [MLXObservabilityEvent] = []
    private var recentRequests: [MLXRequestSummary] = []

    var configuration: MLXObservabilityConfiguration {
        lock.lock()
        let value = currentConfiguration
        lock.unlock()
        return value
    }

    init() {
        lock.name = "org.mlxfoundationmodel.observability"
    }

    func configure(
        _ configuration: MLXObservabilityConfiguration,
        sink: (any MLXObservabilitySink)?
    ) {
        lock.lock()
        currentConfiguration = configuration
        currentSink = sink
        trimLocked()
        lock.unlock()
    }

    func reset(keepingConfiguration: Bool) {
        lock.lock()
        counters.removeAll(keepingCapacity: true)
        gauges.removeAll(keepingCapacity: true)
        histograms.removeAll(keepingCapacity: true)
        recentEvents.removeAll(keepingCapacity: true)
        recentRequests.removeAll(keepingCapacity: true)
        if !keepingConfiguration {
            currentConfiguration = .default
            currentSink = nil
        }
        lock.unlock()
    }

    func snapshot() -> MLXObservabilitySnapshot {
        lock.lock()
        let histogramSnapshot = histograms.mapValues(\.summary)
        let snapshot = MLXObservabilitySnapshot(
            counters: counters,
            gauges: gauges,
            histograms: histogramSnapshot,
            recentEvents: recentEvents,
            recentRequests: recentRequests
        )
        lock.unlock()
        return snapshot
    }

    func incrementCounter(_ name: String, by amount: Double) {
        lock.lock()
        guard currentConfiguration.isEnabled else {
            lock.unlock()
            return
        }
        counters[name, default: 0] += amount
        lock.unlock()
    }

    func setGauge(_ name: String, value: Double) {
        lock.lock()
        guard currentConfiguration.isEnabled else {
            lock.unlock()
            return
        }
        gauges[name] = value
        lock.unlock()
    }

    func recordHistogram(_ name: String, value: Double) {
        lock.lock()
        guard currentConfiguration.isEnabled else {
            lock.unlock()
            return
        }
        histograms[name, default: MLXHistogramAccumulator()].record(value)
        lock.unlock()
    }

    func record(_ event: MLXObservabilityEvent) {
        let configuration: MLXObservabilityConfiguration
        let sink: (any MLXObservabilitySink)?
        lock.lock()
        configuration = currentConfiguration
        guard configuration.isEnabled else {
            lock.unlock()
            return
        }
        sink = currentSink
        if configuration.keptRecentEventCount > 0 {
            recentEvents.append(event)
            trimEventsLocked(configuration.keptRecentEventCount)
        }
        lock.unlock()

        sink?.record(event)
        logEvent(event, configuration: configuration)
    }

    func recordRequestSummary(_ summary: MLXRequestSummary) {
        let configuration: MLXObservabilityConfiguration
        let sink: (any MLXObservabilitySink)?
        lock.lock()
        configuration = currentConfiguration
        guard configuration.isEnabled else {
            lock.unlock()
            return
        }
        sink = currentSink
        recordRequestSummaryCountersLocked(summary)
        recordRequestSummaryHistogramsLocked(summary)
        retainRequestSummaryLocked(summary, configuration: configuration)
        let event = Self.requestSummaryEvent(summary)
        retainEventLocked(event, configuration: configuration)
        lock.unlock()

        sink?.recordRequest(summary)
        sink?.record(event)
        logEvent(event, configuration: configuration)
    }

    private func recordRequestSummaryCountersLocked(_ summary: MLXRequestSummary) {
        counters["generation.requests", default: 0] += 1
        counters["generation.tokens.prompt", default: 0] += Double(summary.promptTokens)
        counters["generation.tokens.generated", default: 0] += Double(summary.generatedTokens)
        counters["generation.tokens.total", default: 0] += Double(summary.totalTokens)
        if let cachedPromptTokens = summary.cachedPromptTokens {
            counters["prompt_cache.tokens.reused", default: 0] += Double(cachedPromptTokens)
        }
    }

    private func recordRequestSummaryHistogramsLocked(_ summary: MLXRequestSummary) {
        histograms["generation.duration_seconds", default: MLXHistogramAccumulator()]
            .record(summary.totalDurationSeconds)
        if let timeToFirstTokenSeconds = summary.timeToFirstTokenSeconds {
            histograms["generation.time_to_first_token_seconds", default: MLXHistogramAccumulator()]
                .record(timeToFirstTokenSeconds)
        }
        if let generationTokensPerSecond = summary.generationTokensPerSecond {
            histograms["generation.tokens_per_second", default: MLXHistogramAccumulator()]
                .record(generationTokensPerSecond)
            histograms["generation.generation_tokens_per_second", default: MLXHistogramAccumulator()]
                .record(generationTokensPerSecond)
        }
        if let promptTokensPerSecond = summary.promptTokensPerSecond {
            histograms["generation.prompt_tokens_per_second", default: MLXHistogramAccumulator()]
                .record(promptTokensPerSecond)
        }
        if let totalTokensPerSecond = summary.totalTokensPerSecond {
            histograms["generation.total_tokens_per_second", default: MLXHistogramAccumulator()]
                .record(totalTokensPerSecond)
        }
    }

    private func retainRequestSummaryLocked(
        _ summary: MLXRequestSummary,
        configuration: MLXObservabilityConfiguration
    ) {
        if configuration.keptRecentRequestCount > 0 {
            recentRequests.append(summary)
            trimRequestsLocked(configuration.keptRecentRequestCount)
        }
    }

    private func retainEventLocked(
        _ event: MLXObservabilityEvent,
        configuration: MLXObservabilityConfiguration
    ) {
        if configuration.keptRecentEventCount > 0 {
            recentEvents.append(event)
            trimEventsLocked(configuration.keptRecentEventCount)
        }
    }

    private static func requestSummaryEvent(_ summary: MLXRequestSummary) -> MLXObservabilityEvent {
        MLXObservabilityEvent(
            name: "generation.request_summary",
            kind: .requestSummary,
            category: .generation,
            severity: .info,
            attributes: [
                "request_id": summary.requestID.uuidString,
                "model": summary.modelName ?? "unknown",
                "strategy": summary.strategy ?? "unknown",
                "stop_reason": summary.stopReason
            ],
            measurements: requestSummaryMeasurements(summary)
        )
    }

    private static func requestSummaryMeasurements(_ summary: MLXRequestSummary) -> [String: Double] {
        var measurements = [
            "prompt_tokens": Double(summary.promptTokens),
            "generated_tokens": Double(summary.generatedTokens),
            "total_tokens": Double(summary.totalTokens),
            "duration_seconds": summary.totalDurationSeconds,
            "tokens_per_second": summary.generationTokensPerSecond ?? 0
        ]
        measurements["prompt_processing_seconds"] = summary.promptProcessingSeconds
        measurements["time_to_first_token_seconds"] = summary.timeToFirstTokenSeconds
        measurements["prompt_tokens_per_second"] = summary.promptTokensPerSecond
        measurements["generation_tokens_per_second"] = summary.generationTokensPerSecond
        measurements["total_tokens_per_second"] = summary.totalTokensPerSecond
        return measurements
    }

    private func trimLocked() {
        trimEventsLocked(currentConfiguration.keptRecentEventCount)
        trimRequestsLocked(currentConfiguration.keptRecentRequestCount)
    }

    private func trimEventsLocked(_ limit: Int) {
        if limit <= 0 {
            recentEvents.removeAll(keepingCapacity: true)
        } else if recentEvents.count > limit {
            recentEvents.removeFirst(recentEvents.count - limit)
        }
    }

    private func trimRequestsLocked(_ limit: Int) {
        if limit <= 0 {
            recentRequests.removeAll(keepingCapacity: true)
        } else if recentRequests.count > limit {
            recentRequests.removeFirst(recentRequests.count - limit)
        }
    }

    private func logEvent(
        _ event: MLXObservabilityEvent,
        configuration: MLXObservabilityConfiguration
    ) {
        guard configuration.osLogEnabled,
              event.severity >= configuration.minimumLogSeverity else {
            return
        }

        let logger = Logger(
            subsystem: configuration.subsystem,
            category: event.category.rawValue
        )
        let attributes = Self.format(event.attributes)
        let measurements = Self.format(event.measurements)
        let duration = event.durationSeconds.map { String(format: "%.6f", $0) } ?? ""
        switch event.severity {
        case .debug:
            logger.debug(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        case .info:
            logger.info(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        case .notice:
            logger.notice(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        case .warning:
            logger.warning(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        case .error:
            logger.error(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        case .fault:
            logger.fault(
                "\(event.name, privacy: .public) \(attributes, privacy: .public) \(measurements, privacy: .public) duration=\(duration, privacy: .public)"
            )
        }
    }

    private static func format(_ attributes: [String: String]) -> String {
        attributes
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\($0.value)" }
            .joined(separator: " ")
    }

    private static func format(_ measurements: [String: Double]) -> String {
        measurements
            .sorted { $0.key < $1.key }
            .map { "\($0.key)=\(String(format: "%.4f", $0.value))" }
            .joined(separator: " ")
    }
}

internal extension Duration {
    var mlxSeconds: Double {
        Double(components.seconds) + Double(components.attoseconds) / 1e18
    }
}
