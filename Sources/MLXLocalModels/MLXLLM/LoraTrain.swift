import Foundation
import MLX
import MLXNN
import MLXOptimizers
import Tokenizers

struct LoRABatchIterator: Sequence, IteratorProtocol {
    private let dataset: [String]
    private let batchSize: Int
    private let tokenizer: Tokenizer
    private let repeats: Bool

    private var order: [Int]
    private var cursor = 0

    init(dataset: [String], tokenizer: Tokenizer, batchSize: Int, train: Bool) {
        precondition(batchSize > 0, "LoRA batch size must be positive")

        self.dataset = dataset
        self.batchSize = batchSize
        self.tokenizer = tokenizer
        self.repeats = train
        self.order = Array(dataset.indices)

        if train {
            order.shuffle()
        }
    }

    mutating func next() -> (MLXArray, MLXArray, MLXArray)? {
        guard !dataset.isEmpty else {
            return nil
        }

        if cursor >= order.count {
            guard repeats else {
                return nil
            }
            order.shuffle()
            cursor = 0
        }

        let batchEnd = Swift.min(cursor + batchSize, order.count)
        let rows = order[cursor ..< batchEnd].map { tokenizer.encode(text: dataset[$0]) }
        cursor = batchEnd

        return makeCausalLanguageModelBatch(rows)
    }
}

private func makeCausalLanguageModelBatch(_ tokenRows: [[Int]]) -> (MLXArray, MLXArray, MLXArray) {
    let rowCount = tokenRows.count
    let paddedTokenCount = Swift.max(2, tokenRows.map(\.count).max() ?? 0)
    let tokens = MLXArray.zeros([rowCount, paddedTokenCount], type: Int32.self)
    var predictionLengths = [Int32]()
    predictionLengths.reserveCapacity(rowCount)

    for (rowIndex, tokenIDs) in tokenRows.enumerated() {
        if !tokenIDs.isEmpty {
            tokens[rowIndex, 0 ..< tokenIDs.count] = MLXArray(tokenIDs.map(Int32.init))
        }
        predictionLengths.append(Int32(Swift.max(0, tokenIDs.count - 1)))
    }

    return (
        tokens[0..., .stride(to: -1)],
        tokens[0..., 1...],
        MLXArray(predictionLengths)
    )
}

private func replaceLoRALinearLayers(
    layers: LoRALinearLayers,
    transform: (Module) -> Module?
) {
    for (parent, keys) in layers {
        let children = parent.children()
        var replacements = ModuleChildren()

        for key in keys {
            guard let child = children[key], case .value(let module) = child else {
                continue
            }
            guard let replacement = transform(module) else {
                continue
            }
            replacements[key] = .value(replacement)
        }

        if !replacements.isEmpty {
            parent.update(modules: replacements)
        }
    }
}

private func dequantizedIfRequested(_ module: Module, deQuantize: Bool) -> Module {
    guard deQuantize, let quantized = module as? QuantizedLinear else {
        return module
    }

    return Linear(weight: quantized.dequantizedWeight, bias: quantized.bias)
}

private struct LoRATrainingStats {
    private(set) var losses = [Float]()
    private(set) var tokenCount = 0
    private var startedAt = Date.timeIntervalSinceReferenceDate

    mutating func record(loss: MLXArray, tokens: MLXArray) {
        losses.append(loss.item(Float.self))
        tokenCount += tokens.item(Int.self)
    }

    mutating func snapshotAndReset() -> (loss: Float, iterationsPerSecond: Double, tokensPerSecond: Double) {
        let now = Date.timeIntervalSinceReferenceDate
        let elapsed = Swift.max(now - startedAt, .leastNonzeroMagnitude)
        let averageLoss = MLXArray(losses).mean(stream: .cpu).item(Float.self)
        let snapshot = (
            loss: averageLoss,
            iterationsPerSecond: Double(losses.count) / elapsed,
            tokensPerSecond: Double(tokenCount) / elapsed
        )

        losses.removeAll(keepingCapacity: true)
        tokenCount = 0
        startedAt = Date.timeIntervalSinceReferenceDate
        return snapshot
    }

    mutating func resetTimer() {
        startedAt = Date.timeIntervalSinceReferenceDate
    }
}

internal enum LoRATrain {
    public typealias LoraLossFunction = (Module, MLXArray, MLXArray, MLXArray) -> (
        MLXArray, MLXArray
    )

    public struct Parameters: Sendable {
        public var batchSize = 4
        public var iterations = 1_000
        public var stepsPerReport = 10
        public var stepsPerEval = 100
        public var validationBatches = 10
        public var saveEvery = 100
        public var adapterURL: URL?

        public init(
            batchSize: Int = 4,
            iterations: Int = 1_000,
            stepsPerReport: Int = 10,
            stepsPerEval: Int = 100,
            validationBatches: Int = 10,
            saveEvery: Int = 100,
            adapterURL: URL? = nil
        ) {
            self.batchSize = batchSize
            self.iterations = iterations
            self.stepsPerReport = stepsPerReport
            self.stepsPerEval = stepsPerEval
            self.validationBatches = validationBatches
            self.saveEvery = saveEvery
            self.adapterURL = adapterURL
        }
    }

    public static func convert(model: Module, layers: LoRALinearLayers) {
        model.freeze()
        replaceLoRALinearLayers(layers: layers) { module in
            guard let linear = module as? Linear else {
                return nil
            }
            return LoRALinear.from(linear: linear)
        }
    }

    public static func fuse(
        model: Module,
        layers: LoRALinearLayers,
        deQuantize: Bool = false
    ) {
        replaceLoRALinearLayers(layers: layers) { module in
            guard let lora = module as? LoRALayer else {
                return nil
            }
            return dequantizedIfRequested(lora.fused(), deQuantize: deQuantize)
        }
    }

    public static func loss(
        model: Module,
        inputs: MLXArray,
        targets: MLXArray,
        lengths: MLXArray
    ) -> (MLXArray, MLXArray) {
        guard let languageModel = model as? any LLMModel else {
            preconditionFailure("LoRA loss requires an LLMModel")
        }

        let logits = languageModel(inputs, cache: nil).asType(.float32)
        let positions = MLXArray(0 ..< inputs.dim(1))[.newAxis, 0...]
        let lossMask = positions .< lengths[0..., .newAxis]
        let tokenCount = lossMask.sum()
        let normalizer = maximum(tokenCount, MLXArray(1))
        let loss = (crossEntropy(logits: logits, targets: targets) * lossMask).sum() / normalizer
        return (loss, tokenCount)
    }

    public static func evaluate(
        model: Module,
        dataset: [String],
        loss: LoraLossFunction = loss,
        tokenizer: Tokenizer,
        batchSize: Int,
        batchCount: Int
    ) -> Float {
        var weightedLosses = [Float]()
        var tokenCount = 0
        let batches = LoRABatchIterator(
            dataset: dataset,
            tokenizer: tokenizer,
            batchSize: batchSize,
            train: false
        )

        for (iteration, batch) in batches.enumerated() {
            let (inputs, targets, lengths) = batch
            let (batchLoss, batchTokens) = loss(model, inputs, targets, lengths)
            weightedLosses.append((batchLoss * batchTokens).item(Float.self))
            tokenCount += batchTokens.item(Int.self)

            if batchCount != 0 && iteration + 1 >= batchCount {
                break
            }
        }

        guard tokenCount > 0 else {
            return .nan
        }
        return (sum(MLXArray(weightedLosses), stream: .cpu) / tokenCount).item(Float.self)
    }

    public static func loadLoRAWeights(model: Module, url: URL) throws {
        let weights = try ModuleParameters.unflattened(loadArrays(url: url))
        try model.update(parameters: weights, verify: .noUnusedKeys)
        eval(model)
    }

    public static func saveLoRAWeights(model: Module, url: URL) throws {
        let parameters = Dictionary(uniqueKeysWithValues: model.trainableParameters().flattened())
        try save(arrays: parameters, url: url)
    }

    public enum Progress: CustomStringConvertible, Sendable {
        case train(
            iteration: Int,
            trainingLoss: Float,
            iterationsPerSecond: Double,
            tokensPerSecond: Double
        )
        case validation(iteration: Int, validationLoss: Float, validationTime: Double)
        case save(iteration: Int, url: URL)

        public var description: String {
            switch self {
            case .train(
                let iteration,
                let trainingLoss,
                let iterationsPerSecond,
                let tokensPerSecond
            ):
                "Iteration \(iteration + 1): training loss \(trainingLoss.formatted()), "
                    + "iterations/sec \(iterationsPerSecond.formatted()), "
                    + "Tokens/sec \(tokensPerSecond.formatted())"

            case .validation(let iteration, let validationLoss, let validationTime):
                "Iteration \(iteration + 1): "
                    + "validation loss \(validationLoss.formatted()), "
                    + "validation time \(validationTime.formatted())s"

            case .save(let iteration, let url):
                "Iteration \(iteration + 1): saved weights to \(url.path())"
            }
        }
    }

    public enum ProgressDisposition: Sendable {
        case stop
        case more
    }

    public static func train(
        model: Module,
        train: [String],
        validate: [String],
        optimizer: Optimizer,
        loss: @escaping LoraLossFunction = loss,
        tokenizer: Tokenizer,
        parameters: Parameters,
        progress: (Progress) -> ProgressDisposition
    ) throws {
        let lossValueGrad = valueAndGrad(model: model) { model, arrays in
            let (lossValue, tokens) = loss(model, arrays[0], arrays[1], arrays[2])
            return [lossValue, tokens]
        }
        var stats = LoRATrainingStats()
        let batches = LoRABatchIterator(
            dataset: train,
            tokenizer: tokenizer,
            batchSize: parameters.batchSize,
            train: true
        )

        for (iteration, batch) in batches.enumerated() {
            let (inputs, targets, lengths) = batch
            let (result, gradients) = lossValueGrad(model, [inputs, targets, lengths])
            let batchLoss = result[0]
            let batchTokens = result[1]

            optimizer.update(model: model, gradients: gradients)
            eval(model, optimizer, batchLoss)
            stats.record(loss: batchLoss, tokens: batchTokens)

            if shouldReport(iteration: iteration, every: parameters.stepsPerReport) {
                let snapshot = stats.snapshotAndReset()
                if progress(
                    .train(
                        iteration: iteration,
                        trainingLoss: snapshot.loss,
                        iterationsPerSecond: snapshot.iterationsPerSecond,
                        tokensPerSecond: snapshot.tokensPerSecond
                    )
                ) == .stop {
                    break
                }
            }

            if shouldValidate(iteration: iteration, every: parameters.stepsPerEval) {
                let validationStart = Date.timeIntervalSinceReferenceDate
                let validationLoss = evaluate(
                    model: model,
                    dataset: validate,
                    loss: loss,
                    tokenizer: tokenizer,
                    batchSize: parameters.batchSize,
                    batchCount: parameters.validationBatches
                )
                let validationTime = Date.timeIntervalSinceReferenceDate - validationStart

                if progress(
                    .validation(
                        iteration: iteration,
                        validationLoss: validationLoss,
                        validationTime: validationTime
                    )
                ) == .stop {
                    break
                }
                stats.resetTimer()
            }

            if let adapterURL = parameters.adapterURL,
                shouldSave(iteration: iteration, every: parameters.saveEvery)
            {
                try saveLoRAWeights(model: model, url: adapterURL)
                if progress(.save(iteration: iteration, url: adapterURL)) == .stop {
                    break
                }
                stats.resetTimer()
            }

            if iteration + 1 >= parameters.iterations {
                break
            }
        }
    }

    private static func shouldReport(iteration: Int, every interval: Int) -> Bool {
        interval > 0 && (iteration + 1) % interval == 0
    }

    private static func shouldValidate(iteration: Int, every interval: Int) -> Bool {
        iteration == 0 || (interval > 0 && (iteration + 1) % interval == 0)
    }

    private static func shouldSave(iteration: Int, every interval: Int) -> Bool {
        interval > 0 && (iteration + 1) % interval == 0
    }
}
