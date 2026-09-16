import Foundation

/// Checkpoint identity is independent of the architecture that executes it.
public enum CheckpointIdentity {
    public static let swiftQwen38 = "swift-qwen3.8-27b-int4g64"

    static func qwenMTPEnabled(modelID: String, setting: String?) -> Bool {
        modelID == swiftQwen38 ? setting == "1" : setting != "0"
    }
}
