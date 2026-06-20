import Foundation

/// Errors raised while building an oQ plan from model artifacts.
public enum MLXOQModelArtifactPlannerError: Error, Equatable, Sendable {
    case emptySafetensorsHeader(URL)
    case invalidOQLevel(String)
    case invalidSafetensorsHeader(URL)
    case missingConfig(URL)
    case noSafetensorsTensors(URL)
}
