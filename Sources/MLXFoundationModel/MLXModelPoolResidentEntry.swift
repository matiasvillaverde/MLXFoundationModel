import Foundation
import MLXLocalModels

struct MLXModelPoolResidentEntry {
    let model: MLXLanguageModel
    let session: any MLXGeneratingSession
    let estimatedResidentBytes: Int
    var lastAccess: Date
    var leaseCount: Int = 0
    var pendingUnloadAfterLease = false

    var isPinned: Bool {
        model.runtime.isPinned
    }

    var isEvictable: Bool {
        !isPinned && leaseCount == 0
    }
}
