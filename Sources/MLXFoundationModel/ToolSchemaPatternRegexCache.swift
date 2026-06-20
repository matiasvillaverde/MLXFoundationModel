import Foundation

final class ToolSchemaPatternRegexCache: @unchecked Sendable {
    private let lock = NSLock()
    private var expressions: [String: NSRegularExpression] = [:]

    deinit {
        // Required by strict lint; cached expressions do not need manual cleanup.
    }

    func expression(for pattern: String) -> NSRegularExpression? {
        lock.lock()
        defer {
            lock.unlock()
        }
        if let cached = expressions[pattern] {
            return cached
        }
        guard let expression = try? NSRegularExpression(pattern: pattern) else {
            return nil
        }
        expressions[pattern] = expression
        return expression
    }
}
