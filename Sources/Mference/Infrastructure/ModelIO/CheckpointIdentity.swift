import Foundation

/// Checkpoint identity is independent of the architecture that executes it.
public enum CheckpointIdentity {
    public static let baseQwen38 = "qwen3.8-27b-4bit"

    static func qwenMTPEnabled(modelID: String, setting: String?) -> Bool {
        setting != "0"
    }
}
