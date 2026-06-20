import Foundation
import MLXLocalModels

struct MLXModelPoolLoadingEntry {
    let id: UUID
    let model: MLXLanguageModel
    let session: any MLXGeneratingSession
    let estimatedResidentBytes: Int
    let startedAt: Date
    let task: Task<Void, any Error>
    var pendingUnloadAfterLoad = false
}
