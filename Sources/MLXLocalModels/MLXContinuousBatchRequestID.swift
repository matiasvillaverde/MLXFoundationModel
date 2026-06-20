internal struct MLXContinuousBatchRequestID: Hashable, Comparable, Sendable, CustomStringConvertible,
    ExpressibleByIntegerLiteral {
    internal let rawValue: Int

    internal init(_ rawValue: Int) {
        self.rawValue = rawValue
    }

    internal init(integerLiteral value: Int) {
        self.rawValue = value
    }

    internal static func < (
        lhs: Self,
        rhs: Self
    ) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    internal var description: String {
        String(rawValue)
    }
}
