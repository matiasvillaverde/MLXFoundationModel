enum MLXSpecPrefillChunkSelector {
    static func selectedTokenIndices(
        importance: [Float],
        keepRate: Double = 0.3,
        chunkSize: Int = 32
    ) -> [Int] {
        guard !importance.isEmpty else {
            return []
        }
        let chunkSize = max(1, chunkSize)
        let keepRate = normalizedKeepRate(keepRate)
        if keepRate >= 1 {
            return Array(importance.indices)
        }

        let chunkCount = (importance.count + chunkSize - 1) / chunkSize
        let keepCount = max(1, Int((Double(chunkCount) * keepRate).rounded(.up)))
        let retainedChunks = topChunks(
            importance: importance,
            chunkSize: chunkSize,
            keepCount: keepCount
        )
        return retainedChunks
            .sorted()
            .flatMap { chunkIndex in
                let start = chunkIndex * chunkSize
                let end = min(start + chunkSize, importance.count)
                return start..<end
            }
    }

    private static func normalizedKeepRate(_ value: Double) -> Double {
        guard value.isFinite else {
            return 0.3
        }
        return min(1, max(0, value))
    }

    private static func topChunks(
        importance: [Float],
        chunkSize: Int,
        keepCount: Int
    ) -> [Int] {
        var heap = ChunkScoreHeap(minimumCapacity: keepCount)
        for chunkIndex in 0..<((importance.count + chunkSize - 1) / chunkSize) {
            let score = chunkScore(
                importance: importance,
                chunkIndex: chunkIndex,
                chunkSize: chunkSize
            )
            heap.insertKeepingBest(ChunkScore(index: chunkIndex, score: score))
        }
        return heap.values.map(\.index)
    }

    private static func chunkScore(
        importance: [Float],
        chunkIndex: Int,
        chunkSize: Int
    ) -> Float {
        let start = chunkIndex * chunkSize
        let end = min(start + chunkSize, importance.count)
        var total: Float = 0
        for index in start..<end {
            total += importance[index]
        }
        return total / Float(end - start)
    }
}

private struct ChunkScore: Equatable {
    let index: Int
    let score: Float

    func isBetter(than other: Self) -> Bool {
        if score == other.score {
            return index < other.index
        }
        return score > other.score
    }

    func isWorse(than other: Self) -> Bool {
        other.isBetter(than: self)
    }
}

private struct ChunkScoreHeap {
    private let minimumCapacity: Int
    private var storage: [ChunkScore] = []

    var values: [ChunkScore] {
        storage
    }

    init(minimumCapacity: Int) {
        self.minimumCapacity = max(1, minimumCapacity)
        storage.reserveCapacity(self.minimumCapacity)
    }

    mutating func insertKeepingBest(_ value: ChunkScore) {
        if storage.count < minimumCapacity {
            storage.append(value)
            siftUp(storage.count - 1)
            return
        }
        guard let currentWorst = storage.first,
              value.isBetter(than: currentWorst) else {
            return
        }
        storage[0] = value
        siftDown(0)
    }

    private mutating func siftUp(_ index: Int) {
        var child = index
        while child > 0 {
            let parent = (child - 1) / 2
            guard storage[child].isWorse(than: storage[parent]) else {
                break
            }
            storage.swapAt(child, parent)
            child = parent
        }
    }

    private mutating func siftDown(_ index: Int) {
        var parent = index
        while true {
            let left = parent * 2 + 1
            let right = left + 1
            var candidate = parent
            if left < storage.count, storage[left].isWorse(than: storage[candidate]) {
                candidate = left
            }
            if right < storage.count, storage[right].isWorse(than: storage[candidate]) {
                candidate = right
            }
            guard candidate != parent else {
                return
            }
            storage.swapAt(parent, candidate)
            parent = candidate
        }
    }
}
