import Foundation

/// The models a machine's memory rules out, before any load (L4, spec 4.3).
/// Platform-neutral so the model library and picker build on the iPhone too;
/// `EngineService` applies the same rule at load.
public enum EngineMemoryGate {
    /// Deep needs more than 16 GB of memory; the reason when `model` may not
    /// run with `physicalMemory`, else nil.
    public static func reason(_ model: AppModel, physicalMemory: UInt64) -> String? {
        model == .deep && physicalMemory <= 16 << 30 ? "Needs a Mac with more than 16 GB of memory." : nil
    }
}
