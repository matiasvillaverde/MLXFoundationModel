import Foundation

extension ToolSchema {
    static func requiredPropertyNames(from schema: JSONObject) -> [String] {
        guard let values = schema["required"] as? [Any] else {
            return []
        }
        var seen: Set<String> = []
        return values.compactMap { value in
            guard let key = value as? String,
                !key.isEmpty,
                seen.insert(key).inserted else {
                return nil
            }
            return key
        }
    }

    static func requiredOrderPermutations<Value>(
        _ values: [Value],
        maximumCount: Int = 720
    ) -> [[Value]] {
        guard values.count > 1 else {
            return [values]
        }
        guard boundedFactorial(values.count, limit: maximumCount) <= maximumCount else {
            return [values]
        }
        return permutations(values)
    }

    private static func boundedFactorial(_ count: Int, limit: Int) -> Int {
        var product = 1
        for value in 2 ... max(2, count) {
            product *= value
            if product > limit {
                return product
            }
        }
        return product
    }

    private static func permutations<Value>(_ values: [Value]) -> [[Value]] {
        var result: [[Value]] = []
        var used = Array(repeating: false, count: values.count)
        var current: [Value] = []
        current.reserveCapacity(values.count)
        appendPermutations(
            values,
            used: &used,
            current: &current,
            result: &result
        )
        return result
    }

    private static func appendPermutations<Value>(
        _ values: [Value],
        used: inout [Bool],
        current: inout [Value],
        result: inout [[Value]]
    ) {
        guard current.count != values.count else {
            result.append(current)
            return
        }
        for index in values.indices where !used[index] {
            used[index] = true
            current.append(values[index])
            appendPermutations(values, used: &used, current: &current, result: &result)
            current.removeLast()
            used[index] = false
        }
    }
}
