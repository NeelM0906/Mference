import Foundation
import MferenceAppCore
import Testing

/// Locks `App/Resources/Info.plist` (copied into Mference.app by
/// `Scripts/bundle.sh`) to the bundle identity of spec P5 and to the
/// `AppPermission.reason` wording, so the system prompt and the broker's
/// one-line reason never drift apart.
@Suite struct ShellBundleTests {
    static let resources = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Resources")

    static func plist(_ name: String) throws -> [String: Any] {
        let data = try Data(contentsOf: resources.appendingPathComponent(name))
        return try #require(PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any])
    }

    @Test func bundleIdentity() throws {
        let info = try Self.plist("Info.plist")
        #expect(info["CFBundleIdentifier"] as? String == "ai.mference.app")
        #expect(info["CFBundleName"] as? String == "Mference")
        #expect(info["CFBundleExecutable"] as? String == "Mference")
        #expect(info["LSMinimumSystemVersion"] as? String == "26.0")
        #expect(info["CFBundlePackageType"] as? String == "APPL")
    }

    @Test func usageStringsMatchPermissionReasons() throws {
        let info = try Self.plist("Info.plist")
        let keys: [AppPermission: [String]] = [
            .calendar: ["NSCalendarsFullAccessUsageDescription", "NSCalendarsUsageDescription"],
            .reminders: ["NSRemindersFullAccessUsageDescription", "NSRemindersUsageDescription"],
            .camera: ["NSCameraUsageDescription"],
            .mailAutomation: ["NSAppleEventsUsageDescription"],
            .microphone: ["NSMicrophoneUsageDescription"],
            .systemAudio: ["NSAudioCaptureUsageDescription"],
            .fullDiskAccess: [],
        ]
        #expect(Set(keys.keys) == Set(AppPermission.allCases), "every permission needs a usage-string decision")
        for (permission, plistKeys) in keys {
            for key in plistKeys {
                #expect(info[key] as? String == permission.reason, "\(key) != AppPermission.\(permission).reason")
            }
        }
    }

    @Test func entitlementsHaveNoSandbox() throws {
        let entitlements = try Self.plist("Mference.entitlements")
        #expect(entitlements["com.apple.security.app-sandbox"] as? Bool == false)
    }
}
