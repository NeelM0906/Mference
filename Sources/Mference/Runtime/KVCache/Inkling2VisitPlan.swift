import Foundation

/// One of the 48 layer visits of an Inkling 2 Mini forward pass: 24 stored
/// layers, layers 6 ..< 18 run three times.
public struct Inkling2Visit: Sendable, Equatable, Hashable {
    /// The visit's own cache slot (KV and convolution state):
    /// `layer` in the first pass (and for the prelude and coda), then
    /// `24 + (pass - 1) * 12 + (layer - 6)` for loop passes 2 and 3 — the
    /// numbering of transformers' `cache_offset` and llama.cpp's loop slots.
    public let slot: Int
    public let layer: Int
    /// 0-based pass; nonzero only for loop-core layers.
    public let pass: Int
    /// Layers 5, 11, 17 and 23 attend globally; the rest use the window.
    public let isGlobal: Bool
}

/// The visit layout of Inkling 2 Mini.
public enum Inkling2VisitPlan {
    public static let storedLayers = 24
    public static let loopStart = 6
    public static let loopEnd = 18
    public static let passes = 3
    public static let loopLayers = loopEnd - loopStart
    public static let visitCount = storedLayers + (passes - 1) * loopLayers

    public static func isGlobal(layer: Int) -> Bool { (layer + 1) % 6 == 0 }

    public static func isLoopLayer(_ layer: Int) -> Bool { (loopStart..<loopEnd).contains(layer) }

    public static func slot(layer: Int, pass: Int) -> Int {
        precondition((0..<storedLayers).contains(layer))
        precondition(pass == 0 || (isLoopLayer(layer) && pass < passes))
        return pass == 0 ? layer : storedLayers + (pass - 1) * loopLayers + (layer - loopStart)
    }

    /// Every visit indexed by its cache slot.
    public static let bySlot: [Inkling2Visit] = (0..<visitCount).map { slot in
        if slot < storedLayers {
            return Inkling2Visit(slot: slot, layer: slot, pass: 0, isGlobal: isGlobal(layer: slot))
        }
        let pass = 1 + (slot - storedLayers) / loopLayers
        let layer = loopStart + (slot - storedLayers) % loopLayers
        return Inkling2Visit(slot: slot, layer: layer, pass: pass, isGlobal: isGlobal(layer: layer))
    }

    /// Visits in execution order: prelude 0 ..< 6, the core three times, coda 18 ..< 24.
    public static let executionOrder: [Inkling2Visit] = {
        var order: [Inkling2Visit] = []
        for layer in 0..<loopStart { order.append(bySlot[layer]) }
        for pass in 0..<passes {
            for layer in loopStart..<loopEnd { order.append(bySlot[slot(layer: layer, pass: pass)]) }
        }
        for layer in loopEnd..<storedLayers { order.append(bySlot[layer]) }
        return order
    }()

    /// Slots of the window visits (40) and of the global visits (8), ascending.
    public static let windowSlots: [Int] = bySlot.filter { !$0.isGlobal }.map(\.slot)
    public static let globalSlots: [Int] = bySlot.filter(\.isGlobal).map(\.slot)
}
