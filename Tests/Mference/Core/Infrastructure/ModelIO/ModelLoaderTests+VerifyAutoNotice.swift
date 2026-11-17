import Foundation
import Metal
import Testing

@testable import Mference

/// The CLI and server print `Model.verifyAutoFallbackNotice` after a load, so
/// a `--verify auto` fallback to full SHA-256 is visible. It is set only when
/// the load requested `.trustedReceiptWhenValid` and a receipt check failed.
extension ModelLoaderTests {
  @Test func verifyAutoNoticeNamesAMissingReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.integrityFallbackDetail == "\(VerifiedInstallReceiptReader.fileName) is missing")
    #expect(model.verifyAutoFallbackNotice == ModelIntegrityPolicy.autoFallbackNotice(
      detail: "\(VerifiedInstallReceiptReader.fileName) is missing"))
  }

  @Test func verifyAutoNoticeNamesAMovedInstall() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["modelDirectoryPath"] =
        dir.deletingLastPathComponent().appendingPathComponent("other.gturbo").standardizedFileURL.path
    }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.integrityFallbackDetail == "model directory mismatch")
    #expect(model.verifyAutoFallbackNotice != nil)
  }

  @Test func verifyAutoNoticeIsAbsentWithAValidReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
    #expect(model.verifyAutoFallbackNotice == nil)
  }

  /// An explicit `--verify full-sha256` is not a fallback, so it prints no
  /// notice even without a receipt.
  @Test func verifyAutoNoticeIsAbsentUnderExplicitFullSha256() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .fullSha256)
    #expect(model.integrityPolicy == .fullSha256)
    #expect(model.verifyAutoFallbackNotice == nil)
  }
}
