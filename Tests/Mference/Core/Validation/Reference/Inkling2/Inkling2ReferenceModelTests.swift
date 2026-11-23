import Foundation
import Testing
import MferenceValidationSupport

/// Whole-model properties of the Inkling 2 CPU reference on the toy
/// checkpoint: the schedule, chunk invariance (chunked prefill equals
/// decode), the loop's restart rule and per-pass state, snapshots, and that
/// the toy reaches every feature path.
@Suite("Inkling2ReferenceModel")
struct Inkling2ReferenceModelTests {

    private func toyModel(_ weights: Inkling2ReferenceWeights = Inkling2ToyCheckpoint.weights()) throws -> Inkling2ReferenceModel {
        try Inkling2ReferenceModel(config: .toy, weights: weights)
    }

    // MARK: Schedule

    @Test func miniScheduleHas48VisitsInExecutionOrder() throws {
        let c = Inkling2ReferenceConfig.mini
        try c.validate()
        let s = c.schedule
        #expect(c.numVisits == 48 && s.count == 48)
        #expect(s.map(\.order) == Array(0..<48))
        #expect(s.map(\.layer) == Array(0..<18) + Array(6..<18) + Array(6..<18) + Array(18..<24))
        #expect(s.map(\.pass) == [Int](repeating: 0, count: 18) + [Int](repeating: 1, count: 12)
                + [Int](repeating: 2, count: 12) + [Int](repeating: 0, count: 6))
        // Pass 2 uses cache slots 24 to 35, pass 3 uses 36 to 47, every slot once.
        #expect(s[18..<30].map(\.cacheSlot) == Array(24..<36))
        #expect(s[30..<42].map(\.cacheSlot) == Array(36..<48))
        #expect(Set(s.map(\.cacheSlot)) == Set(0..<48))
        // Eight global visits: layers 5 and 23 once, 11 and 17 three times.
        #expect(s.filter { !c.slidingLayers[$0.layer] }.map(\.layer) == [5, 11, 17, 11, 17, 11, 17, 23])
        #expect(c.qDim == 3072 && c.kvDim == 1024 && c.headsPerKVHead == 3)
    }

    @Test func toyKeepsMinisScheduleAndValidates() throws {
        let toy = Inkling2ReferenceConfig.toy
        let mini = Inkling2ReferenceConfig.mini
        #expect(toy.schedule == mini.schedule)
        #expect(toy.slidingLayers == mini.slidingLayers)
        try Inkling2ToyCheckpoint.weights().validate(against: toy)
    }

    @Test func validationNamesAMissingTensor() {
        var w = Inkling2ToyCheckpoint.weights()
        w.layers[11].kConvProj = nil
        let missing = #expect(throws: (any Error).self) { try w.validate(against: .toy) }
        #expect(missing?.localizedDescription.contains("layer 11") == true)
        var w2 = Inkling2ToyCheckpoint.weights()
        w2.layers[3].attnResQueries.append(w2.layers[3].ffnResQuery)
        let extra = #expect(throws: (any Error).self) { try w2.validate(against: .toy) }
        #expect(extra?.localizedDescription.contains("attention-residual queries") == true)
    }

    @Test func slidingLayersUseTheWindowAsTheirBiasExtent() {
        // transformers: rel_extent = sliding_window_size on sliding layers,
        // rel_extent on global ones. The toy gives them different values.
        let toy = Inkling2ReferenceConfig.toy
        #expect(toy.relativeExtent(layer: 4) == 8 && toy.relativeExtent(layer: 5) == 12)
        #expect(toy.relativeExtent(layer: 6) == 8 && toy.relativeExtent(layer: 23) == 12)
        let w = Inkling2ToyCheckpoint.weights()
        #expect(w.layers[0].relBias.count == 8 * toy.relDim)
        #expect(w.layers[17].relBias.count == 12 * toy.relDim)
        #expect(Inkling2ReferenceConfig.mini.relativeExtent(layer: 0) == 1024)
        #expect(Inkling2ReferenceConfig.mini.relativeExtent(layer: 11) == 1024)
    }

    // MARK: Determinism and chunking

    @Test func weightsAndLogitsAreDeterministic() throws {
        let tokens = Inkling2ToyCheckpoint.tokens(count: 4)
        let a = try toyModel().forward(tokens: tokens, logits: .all)
        let b = try toyModel().forward(tokens: tokens, logits: .all)
        #expect(a == b)
        #expect(a.count == 4 && a[0].count == Inkling2ReferenceConfig.toy.vocabUnpadded)
        #expect(a.allSatisfy { $0.allSatisfy(\.isFinite) })
        let model = try toyModel()
        model.forward(tokens: tokens)
        model.reset()
        #expect(model.position == 0)
        #expect(model.forward(tokens: tokens, logits: .all) == a)
    }

    @Test func chunkSizeDoesNotChangeTheResult() throws {
        // 15 tokens cross the window (8), the bias extent (12) and the log floor (4).
        let tokens = Inkling2ToyCheckpoint.tokens(count: 15)
        let whole = try toyModel().forward(tokens: tokens, logits: .all)

        let stepped = try toyModel()
        var steps: [[Float]] = []
        for t in tokens { steps += stepped.forward(tokens: [t], logits: .all) }

        let chunked = try toyModel()
        var parts: [[Float]] = []
        var start = 0
        for size in [5, 1, 6, 3] {
            parts += chunked.forward(tokens: Array(tokens[start..<(start + size)]), logits: .all)
            start += size
        }
        // Each token runs the same operations in the same order whatever the
        // chunking, so the results are identical, not just close.
        #expect(steps == whole)
        #expect(parts == whole)
        #expect(chunked.position == 15)
    }

    @Test func everyVisitKeepsItsOwnCache() throws {
        let model = try toyModel()
        model.forward(tokens: Inkling2ToyCheckpoint.tokens(count: 3), logits: .none)
        let state = model.saveState()
        #expect(state.slots.count == 48)
        #expect(state.slots.allSatisfy { $0.keys.count == 3 && $0.values.count == 3 })
        // Pass 1 and pass 2 of a core layer hold different keys.
        #expect(state.slots[6].keys != state.slots[24].keys)
        #expect(state.slots[24].keys != state.slots[36].keys)
    }

    @Test func snapshotRestoreContinuesIdentically() throws {
        let model = try toyModel()
        let prefix = Inkling2ToyCheckpoint.tokens(count: 10)
        let suffix = Inkling2ToyCheckpoint.tokens(count: 4, seed: 99)
        model.forward(tokens: prefix, logits: .none)
        let saved = model.saveState()
        let first = model.forward(tokens: suffix, logits: .all)
        model.restoreState(saved)
        #expect(model.position == 10)
        #expect(model.forward(tokens: suffix, logits: .all) == first)
    }

    // MARK: The loop

    @Test func silentGatesMakeEveryPassTheSameComputation() throws {
        // Each pass restarts from the prelude output with fresh, separate
        // state, so with the message gates silenced all three passes agree
        // exactly, visit by visit. Sharing state across passes, or carrying a
        // pass's output into the next, would break this.
        let model = try toyModel(Inkling2ToyCheckpoint.silencingMessageGates(Inkling2ToyCheckpoint.weights()))
        model.captureTrace = true
        model.forward(tokens: Inkling2ToyCheckpoint.tokens(count: 6), logits: .none)
        let trace = try #require(model.lastTrace)
        #expect(trace.visitOutputs.count == 48)
        for i in 0..<12 {
            #expect(trace.visitOutputs[6 + i] == trace.visitOutputs[18 + i])
            #expect(trace.visitOutputs[6 + i] == trace.visitOutputs[30 + i])
            #expect(trace.routes[6 + i] == trace.routes[30 + i])
        }
    }

    @Test func messagesChangeTheLaterPasses() throws {
        let model = try toyModel()
        model.captureTrace = true
        model.forward(tokens: Inkling2ToyCheckpoint.tokens(count: 6), logits: .none)
        let trace = try #require(model.lastTrace)
        // Pass 1 has no messages, so it matches the silenced run; passes 2 and 3 do not.
        let silent = try toyModel(Inkling2ToyCheckpoint.silencingMessageGates(Inkling2ToyCheckpoint.weights()))
        silent.captureTrace = true
        silent.forward(tokens: Inkling2ToyCheckpoint.tokens(count: 6), logits: .none)
        let silentTrace = try #require(silent.lastTrace)
        #expect(Array(trace.visitOutputs[0..<18]) == Array(silentTrace.visitOutputs[0..<18]))
        #expect(trace.visitOutputs[18] != silentTrace.visitOutputs[18])
        #expect(trace.visitOutputs[30] != trace.visitOutputs[18])
        #expect(trace.finalHidden != silentTrace.finalHidden)
    }

    // MARK: Feature coverage of the toy

    @Test func toyExercisesEveryFeaturePath() throws {
        let model = try toyModel()
        let n = 15
        model.forward(tokens: Inkling2ToyCheckpoint.tokens(count: n), logits: .all)
        let cov = model.coverage
        let c = Inkling2ReferenceConfig.toy
        let headVisits = n * c.numVisits * c.numHeads
        let kvVisits = n * c.numVisits * c.numKVHeads
        let seen = Comment(rawValue: "\(cov)")
        // Clips: active for some head vectors and not for all.
        #expect(cov.clippedQueries > 0 && cov.clippedQueries < headVisits, seen)
        #expect(cov.clippedKeys > 0 && cov.clippedKeys < kvVisits, seen)
        #expect(cov.clippedRelStates > 0 && cov.clippedRelStates < headVisits, seen)
        // Positions past the floor on global visits.
        #expect(cov.logScaledQueries > 0, seen)
        // Keys past the window on sliding visits, past the extent on global ones.
        #expect(cov.windowMaskedKeys > 0, seen)
        #expect(cov.keysPastBiasExtent > 0, seen)
        // 4 controllers x 24 later-pass visits x tokens.
        #expect(cov.messageGates == 4 * 24 * n, seen)
        #expect(cov.biasChangedRoutes > 0, seen)
        #expect(cov.saturatedGates > 0, seen)
        #expect(cov.saturatedUps > 0, seen)
        #expect(cov.saturatedLogits > 0, seen)
    }

    @Test func routesPickDistinctExpertsWithScaledWeights() throws {
        let model = try toyModel()
        model.captureTrace = true
        model.forward(tokens: Inkling2ToyCheckpoint.tokens(count: 3), logits: .none)
        let trace = try #require(model.lastTrace)
        let c = Inkling2ReferenceConfig.toy
        var used = Set<Int>()
        for visit in trace.routes {
            for r in visit {
                #expect(r.experts.count == c.topK && Set(r.experts).count == c.topK)
                #expect(abs(r.weights.reduce(0, +) - c.routeScale) < 1e-4)
                used.formUnion(r.experts)
            }
        }
        #expect(used.count == c.numExperts)
    }
}
