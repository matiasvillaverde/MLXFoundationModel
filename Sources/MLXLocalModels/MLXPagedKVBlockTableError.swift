internal enum MLXPagedKVBlockTableError: Error, Equatable {
    case duplicateBlockID(MLXPagedKVBlockID)
    case missingBlockID(MLXPagedKVBlockID)
    case missingRowBlock(rowID: MLXGenerationBatchRowID, blockID: MLXPagedKVBlockID)
    case noEvictableBlock(capacity: Int)
    case releaseUnderflow(MLXPagedKVBlockID)
}
