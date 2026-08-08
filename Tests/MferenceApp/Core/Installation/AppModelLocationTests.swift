import Foundation
import Testing
@testable import MferenceAppCore

@Suite struct AppModelLocationTests {
    @Test func rememberedURLBecomesDefaultAcrossLaunches() throws {
        let suiteName = "AppModelLocationTests-\(UUID().uuidString)"
        let userDefaults = try #require(UserDefaults(suiteName: suiteName))
        defer { userDefaults.removePersistentDomain(forName: suiteName) }
        let modelURL = URL(fileURLWithPath: "/models/shared.gturbo", isDirectory: true)

        AppModelLocation.remember(modelURL, userDefaults: userDefaults)

        #expect(AppModelLocation.defaultURL(userDefaults: userDefaults).path
            == "/models/shared.gturbo")
    }

    @Test func explicitURLWins() {
        let result = AppModelLocation.resolve(
            explicitURL: URL(fileURLWithPath: "/models/explicit.gturbo"),
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/models/explicit.gturbo")
    }

    @Test func executableAncestorFindsPackageRootOutsideCWD() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/MferenceApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/repo/.build/debug/MferenceMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/elsewhere"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func currentDirectoryCanBePackageRoot() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/MferenceApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains)
        #expect(result.path == "/repo/scratch/gemma4.gturbo")
    }

    @Test func mapleUsesItsConventionalInstallDirectory() {
        let files: Set<String> = ["/repo/Package.swift", "/repo/Sources/MferenceApp/Mac"]
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/repo"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: files.contains,
            installDirectoryName: AppModelInstallDescriptor.maple.installDirectoryName)
        #expect(result.path == "/repo/scratch/maple.gturbo")
    }

    @Test func standaloneAppFallsBackToApplicationSupport() {
        let result = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: URL(fileURLWithPath: "/Applications/MferenceMac"),
            currentDirectoryURL: URL(fileURLWithPath: "/"),
            applicationSupportURL: URL(fileURLWithPath: "/support"),
            fileExists: { _ in false })
        #expect(result.path == "/support/Mference/gemma4.gturbo")
    }
}
