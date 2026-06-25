import MLX
@testable import MLXLocalModels
import Testing

@Suite("Gated delta architecture")
struct GatedDeltaArchitectureTests {
    @Test("plans grouped value heads and kernel eligibility")
    func plansGroupedValueHeadsAndKernelEligibility() {
        let fallbackLayout = GatedDeltaLayout(
            q: MLXArray.zeros([2, 3, 2, 4]),
            k: MLXArray.zeros([2, 3, 2, 4]),
            v: MLXArray.zeros([2, 3, 4, 5])
        )
        let kernelLayout = GatedDeltaLayout(
            q: MLXArray.zeros([1, 2, 1, 32]),
            k: MLXArray.zeros([1, 2, 1, 32]),
            v: MLXArray.zeros([1, 2, 2, 8])
        )

        #expect(fallbackLayout.valueHeadsPerKeyHead == 2)
        #expect(fallbackLayout.stateShape == [2, 4, 5, 4])
        #expect(!fallbackLayout.supportsMetalKernel)
        #expect(kernelLayout.valueHeadsPerKeyHead == 2)
        #expect(kernelLayout.supportsMetalKernel)
    }

    @Test("decay uses softplus time step")
    func decayUsesSoftplusTimeStep() {
        Device.withDefaultDevice(.cpu) {
            let decay = computeGatedDeltaG(
                MLXArray([Float(0), 0]),
                MLXArray([Float(0), 0]),
                MLXArray([Float(0), 0])
            )
            eval(decay)

            #expect(decay.shape == [2])
            #expect(decay.asArray(Float.self).allSatisfy { abs($0 - 0.5) < 0.0001 })
        }
    }

    @Test("ops fallback updates grouped heads and masks inactive state")
    func opsFallbackUpdatesGroupedHeadsAndMasksInactiveState() {
        Device.withDefaultDevice(.cpu) {
            let queries = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 1, 2])
            let keys = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 1, 2])
            let values = MLXArray([Float(2), 4, 8, 16]).reshaped([1, 2, 2, 1])
            let decay = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 2])
            let beta = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 2])
            let mask = MLXArray([true, false])

            let (output, state) = gatedDeltaOps(
                q: queries,
                k: keys,
                v: values,
                g: decay,
                beta: beta,
                mask: mask
            )
            eval(output, state)

            #expect(output.shape == [1, 2, 2, 1])
            #expect(state.shape == [1, 2, 1, 2])
            #expect(output.asArray(Float.self) == [4, 8, 0, 0])
            #expect(state.asArray(Float.self) == [2, 2, 4, 4])
        }
    }

    @Test("public update falls back for non-kernel key dimensions")
    func publicUpdateFallsBackForNonKernelKeyDimensions() {
        Device.withDefaultDevice(.cpu) {
            let queries = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 1, 2])
            let keys = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 1, 2])
            let values = MLXArray([Float](repeating: 1, count: 4)).reshaped([1, 2, 2, 1])
            let timeSteps = MLXArray([Float](repeating: 0, count: 4)).reshaped([1, 2, 2])
            let betaLogits = MLXArray([Float](repeating: 100, count: 4)).reshaped([1, 2, 2])
            let aLog = MLXArray([Float(0), 0])
            let dtBias = MLXArray([Float](repeating: 100, count: 2))

            let (output, state) = gatedDeltaUpdate(
                q: queries,
                k: keys,
                v: values,
                a: timeSteps,
                b: betaLogits,
                aLog: aLog,
                dtBias: dtBias
            )
            eval(output, state)

            #expect(output.shape == [1, 2, 2, 1])
            #expect(state.shape == [1, 2, 1, 2])
            #expect(all(isFinite(output)).item(Bool.self))
            #expect(all(isFinite(state)).item(Bool.self))
        }
    }
}
