import Foundation

/// Scores calibration observations and returns per-layer oQ sensitivity inputs.
public struct MLXOQCalibrationSensitivityScorer {
    /// Aggregation to use when several tensors from one layer are observed.
    public let aggregation: MLXOQSensitivityAggregation
    private let metric: any MLXOQSensitivityMetric

    /// Creates a scorer with a metric strategy and layer aggregation strategy.
    public init(
        aggregation: MLXOQSensitivityAggregation = .maximum,
        metric: any MLXOQSensitivityMetric = MLXOQRelativeMSESensitivityMetric()
    ) {
        self.aggregation = aggregation
        self.metric = metric
    }

    /// Computes one tensor sensitivity score.
    public func score(
        _ observation: MLXOQCalibrationObservation
    ) throws -> MLXOQLayerSensitivityScore {
        let value = try metric.score(
            reference: observation.referenceOutput,
            candidate: observation.candidateOutput
        )
        guard value.isFinite else {
            throw MLXOQCalibrationSensitivityError.nonFiniteScore(observation.tensorName)
        }
        return MLXOQLayerSensitivityScore(
            layerIndex: observation.layerIndex,
            score: value,
            tensorName: observation.tensorName
        )
    }

    /// Computes per-layer sensitivity scores consumable by `MLXOQQuantizationPlanOptions`.
    public func layerSensitivityScores(
        for observations: [MLXOQCalibrationObservation]
    ) throws -> [Int: Double] {
        let scores = try observations.map(score)
        let groupedScores = Dictionary(grouping: scores, by: \.layerIndex)
        return groupedScores.mapValues { layerScores in
            aggregation.aggregate(layerScores.map(\.score))
        }
    }
}
