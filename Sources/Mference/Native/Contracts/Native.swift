import Foundation

/// Namespace for the native engine (`docs/NATIVE_ENGINE.md`).
///
/// Everything under `Sources/Mference/Native/Contracts/` is a frozen contract:
/// protocols, value types and the normative layout definitions that two or more
/// components must agree on byte for byte. Contracts hold no implementation
/// logic beyond those definitions. After the freeze (`contracts: freeze native
/// engine interfaces v1`) a contract changes only through a coordinator PR that
/// names every affected worker.
///
/// Implementations live beside the contracts: `Native/Reference/` (the slow,
/// frozen oracle), `Native/Weights/`, `Native/Kernels/`, `Native/Memory/`,
/// `Native/Scheduler/`, `Native/Sampling/` and, in Phase 2, `Native/Runner/`.
public enum Native {
    /// Version of the contract set. Bumped only by a coordinator contract PR.
    public static let contractVersion = 1
}

extension Native {
    /// Errors shared across native components. Components may define their own
    /// error enums for local failures; anything that crosses a contract boundary
    /// is one of these.
    public enum NativeError: Error, Equatable, CustomStringConvertible {
        /// A tensor, layer or metadata key the contract requires is missing.
        case missing(String)
        /// Shapes, dtypes, row counts or layouts do not satisfy the contract.
        case invalidShape(String)
        /// The quant type / layout / GPU family combination is not implemented.
        case unsupported(String)
        /// A hash, version or manifest check failed.
        case integrity(String)
        /// A pipeline, buffer or library could not be created.
        case resource(String)
        /// A committed command buffer finished with an error.
        case commandBufferFailed(String)
        /// The memory plan does not fit the budget it was resolved against.
        case memoryBudgetExceeded(required: UInt64, budget: UInt64)
        /// A position, page or snapshot request is outside the stored state.
        case stateOutOfRange(String)
        /// The step or generation was cancelled.
        case cancelled

        public var description: String {
            switch self {
            case .missing(let s): return "native: missing \(s)"
            case .invalidShape(let s): return "native: invalid shape: \(s)"
            case .unsupported(let s): return "native: unsupported: \(s)"
            case .integrity(let s): return "native: integrity check failed: \(s)"
            case .resource(let s): return "native: resource: \(s)"
            case .commandBufferFailed(let s): return "native: command buffer failed: \(s)"
            case .memoryBudgetExceeded(let r, let b):
                return "native: memory plan needs \(r) bytes, budget \(b)"
            case .stateOutOfRange(let s): return "native: state out of range: \(s)"
            case .cancelled: return "native: cancelled"
            }
        }
    }
}
