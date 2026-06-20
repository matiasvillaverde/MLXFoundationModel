import Foundation

struct MLXThinkTagStreamSplitter: Sendable {
    enum SegmentKind: Sendable {
        case response
        case reasoning
    }

    struct Segment: Sendable, Equatable {
        let kind: SegmentKind
        let text: String
    }

    private enum Mode {
        case response
        case reasoning

        var segmentKind: SegmentKind {
            switch self {
            case .response:
                return .response

            case .reasoning:
                return .reasoning
            }
        }
    }

    private enum MarkerAction {
        case openReasoning
        case closeReasoning
    }

    private var mode = Mode.response
    private var buffer = ""
    private var accumulatedReasoning = ""
    private var hasEmittedResponse = false
    private var shouldDropMarkerNewline = false

    init(startInReasoning: Bool = false) {
        mode = startInReasoning ? .reasoning : .response
    }

    var retainedUTF8ByteCount: Int {
        buffer.utf8.count
    }

    mutating func consume(_ text: String) -> [Segment] {
        guard !text.isEmpty else {
            return []
        }
        buffer += text
        return drain(final: false)
    }

    mutating func finish() -> [Segment] {
        drain(final: true)
    }

    private mutating func drain(final: Bool) -> [Segment] {
        var segments: [Segment] = []
        while !buffer.isEmpty {
            dropMarkerNewlineIfNeeded()
            guard let marker = nextMarker() else {
                emitBufferedSuffix(final: final, into: &segments)
                break
            }
            appendSegment(String(buffer[..<marker.range.lowerBound]), to: &segments)
            buffer.removeSubrange(..<marker.range.upperBound)
            apply(marker.action)
            shouldDropMarkerNewline = true
        }
        recoverUnclosedReasoningAsResponse(final: final, into: &segments)
        return coalesced(segments)
    }

    private mutating func emitBufferedSuffix(final: Bool, into segments: inout [Segment]) {
        let keepCount = final ? 0 : partialMarkerSuffixLength(in: buffer)
        let endIndex = buffer.index(buffer.endIndex, offsetBy: -keepCount)
        appendSegment(String(buffer[..<endIndex]), to: &segments)
        buffer = String(buffer[endIndex...])
    }

    private mutating func dropMarkerNewlineIfNeeded() {
        guard shouldDropMarkerNewline else {
            return
        }
        if buffer.hasPrefix("\r\n") {
            buffer.removeFirst(2)
            shouldDropMarkerNewline = false
        } else if buffer.hasPrefix("\n") {
            buffer.removeFirst()
            shouldDropMarkerNewline = false
        } else if !buffer.isEmpty {
            shouldDropMarkerNewline = false
        }
    }

    private mutating func appendSegment(_ text: String, to segments: inout [Segment]) {
        guard !text.isEmpty else {
            return
        }
        segments.append(Segment(kind: mode.segmentKind, text: text))
        switch mode.segmentKind {
        case .reasoning:
            accumulatedReasoning += text

        case .response:
            hasEmittedResponse = true
        }
    }

    private mutating func recoverUnclosedReasoningAsResponse(
        final: Bool,
        into segments: inout [Segment]
    ) {
        guard final,
            mode == .reasoning,
            !hasEmittedResponse,
            !accumulatedReasoning.isEmpty
        else {
            return
        }
        segments.append(Segment(kind: .response, text: accumulatedReasoning))
        hasEmittedResponse = true
    }

    private mutating func apply(_ action: MarkerAction) {
        switch action {
        case .openReasoning:
            mode = .reasoning

        case .closeReasoning:
            mode = .response
        }
    }

    private func nextMarker() -> (range: Range<String.Index>, action: MarkerAction)? {
        activeMarkers.reduce(nil) { best, marker in
            guard let range = buffer.range(of: marker.text) else {
                return best
            }
            let candidate = (range: range, action: marker.action)
            guard let best else {
                return candidate
            }
            return range.lowerBound < best.range.lowerBound ? candidate : best
        }
    }

    private var activeMarkers: [(text: String, action: MarkerAction)] {
        switch mode {
        case .response:
            return [("<think>", .openReasoning)]

        case .reasoning:
            return [
                ("</think>", .closeReasoning),
                ("<think>", .openReasoning)
            ]
        }
    }

    private func partialMarkerSuffixLength(in text: String) -> Int {
        let maximumLength = min(
            text.count,
            max(activeMarkers.map(\.text.count).max() ?? 0, 1) - 1
        )
        guard maximumLength > 0 else {
            return 0
        }
        for length in stride(from: maximumLength, through: 1, by: -1) {
            let suffix = String(text.suffix(length))
            if activeMarkers.contains(where: { $0.text.hasPrefix(suffix) }) {
                return length
            }
        }
        return 0
    }

    private func coalesced(_ segments: [Segment]) -> [Segment] {
        segments.reduce(into: []) { result, segment in
            guard let last = result.last,
                last.kind == segment.kind
            else {
                result.append(segment)
                return
            }
            result[result.count - 1] = Segment(
                kind: last.kind,
                text: last.text + segment.text
            )
        }
    }
}
