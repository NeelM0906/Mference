import Foundation
import Synchronization

/// A `PermissionBroker` for tests and previews: states are set by the test,
/// a request answers with `answer`, and every call is recorded. Never
/// touches TCC.
public final class FakePermissionBroker: PermissionBroker {
    public enum Call: Sendable, Hashable {
        case state(AppPermission)
        case request(AppPermission)
        case openSystemSettings(AppPermission)
    }

    private struct State {
        var states: [AppPermission: PermissionState]
        var answers: [AppPermission: PermissionState]
        var calls: [Call] = []
    }

    private let state: Mutex<State>

    /// - Parameters:
    ///   - states: current states; missing ones are `.notDetermined`.
    ///   - answers: what a request turns a not-determined permission into;
    ///     missing ones are `.granted`.
    public init(states: [AppPermission: PermissionState] = [:],
                answers: [AppPermission: PermissionState] = [:]) {
        state = Mutex(State(states: states, answers: answers))
    }

    public var calls: [Call] { state.withLock { $0.calls } }

    public func setState(_ value: PermissionState, for permission: AppPermission) {
        state.withLock { $0.states[permission] = value }
    }

    public func state(of permission: AppPermission) async -> PermissionState {
        state.withLock { s in
            s.calls.append(.state(permission))
            return s.states[permission] ?? .notDetermined
        }
    }

    public func request(_ permission: AppPermission) async -> PermissionState {
        state.withLock { s in
            s.calls.append(.request(permission))
            let current = s.states[permission] ?? .notDetermined
            guard current == .notDetermined else { return current }
            let answer = s.answers[permission] ?? .granted
            s.states[permission] = answer
            return answer
        }
    }

    public func openSystemSettings(for permission: AppPermission) async {
        state.withLock { $0.calls.append(.openSystemSettings(permission)) }
    }
}
