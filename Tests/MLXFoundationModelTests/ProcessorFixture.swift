@testable import MLXLocalModels

enum ProcessorFixture {
    case empty
    case masked

    var first: LogitProcessor? {
        switch self {
        case .empty:
            nil

        case .masked:
            SuppressTokensProcessor(tokenIds: [1])
        }
    }

    var second: LogitProcessor? {
        switch self {
        case .empty:
            nil

        case .masked:
            SuppressTokensProcessor(tokenIds: [0])
        }
    }
}
