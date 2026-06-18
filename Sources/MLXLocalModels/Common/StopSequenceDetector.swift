import Foundation

/// Incremental detector that withholds possible stop-sequence suffixes.
internal struct StopSequenceDetector: Sendable {
    internal enum Result: Equatable, Sendable {
        case more(String)
        case stop(String)
    }

    private let sequences: [String]
    private let maxSequenceLength: Int
    private var buffer: String = ""

    internal init(sequences: [String]) {
        let filtered = sequences.filter { !$0.isEmpty }
        self.sequences = filtered
        self.maxSequenceLength = filtered.map(\.count).max() ?? 0
    }

    internal mutating func append(_ text: String) -> Result {
        guard !text.isEmpty else { return .more("") }
        guard !sequences.isEmpty else { return .more(text) }

        buffer += text
        if let stopRange = earliestStopRange(in: buffer) {
            let safeText = String(buffer[..<stopRange.lowerBound])
            buffer = ""
            return .stop(safeText)
        }

        let pendingCount = max(0, maxSequenceLength - 1)
        guard buffer.count > pendingCount else { return .more("") }

        let safeEnd = buffer.index(buffer.endIndex, offsetBy: -pendingCount)
        let safeText = String(buffer[..<safeEnd])
        buffer = String(buffer[safeEnd...])
        return .more(safeText)
    }

    internal mutating func flush() -> String {
        let text = buffer
        buffer = ""
        return text
    }

    private func earliestStopRange(in text: String) -> Range<String.Index>? {
        var earliest: Range<String.Index>?
        for sequence in sequences {
            guard let range = text.range(of: sequence) else { continue }
            if let current = earliest {
                if range.lowerBound < current.lowerBound {
                    earliest = range
                }
            } else {
                earliest = range
            }
        }
        return earliest
    }
}
