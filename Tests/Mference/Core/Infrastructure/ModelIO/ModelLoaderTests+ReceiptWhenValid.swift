import Foundation
import Metal
import Testing

@testable import Mference

/// `.trustedReceiptWhenValid` is the CLI and server default: the install
/// receipt's size checks when the receipt validates, full SHA-256 otherwise.
extension ModelLoaderTests {
  @Test func receiptWhenValidUsesAValidReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    // Same-size corruption is what the receipt mode knowingly does not detect;
    // reading the expert without an error shows no hashing took place.
    try Self.flipByte(
      in: dir.appendingPathComponent("packed_experts").appendingPathComponent("layer_00.bin"), at: 64)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .sizeCheckTrustedReceipt)
    _ = try model.routedExpert(layer: 0, expert: 0)
  }

  @Test func receiptWhenValidFallsBackToFullSha256WithoutAReceipt() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.flipByte(
      in: dir.appendingPathComponent("packed_experts").appendingPathComponent("layer_00.bin"), at: 64)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
    #expect {
      _ = try model.routedExpert(layer: 0, expert: 0)
    } throws: { error in
      if case ModelError.checksumMismatch = error { return true }
      return false
    }
  }

  @Test func receiptWhenValidFallsBackToFullSha256WhenTheReceiptDoesNotValidate() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    // A moved or copied install: the receipt names another directory.
    try Self.mutateReceipt(directoryURL: dir) { root in
      root["modelDirectoryPath"] =
        dir.deletingLastPathComponent().appendingPathComponent("other.gturbo").standardizedFileURL.path
    }
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
    _ = try model.routedExpert(layer: 0, expert: 0)
  }

  @Test func receiptWhenValidStillRejectsAWrongSizedLayerFile() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let layerURL = dir.appendingPathComponent("packed_experts").appendingPathComponent("layer_00.bin")
    let handle = try FileHandle(forWritingTo: layerURL)
    try handle.truncate(atOffset: 1024)
    try handle.close()
    let device = try #require(MTLCreateSystemDefaultDevice())

    // The size check fails, so the load falls back to hashing, which rejects
    // the truncated file when it is first touched.
    let model = try Model.load(
      directoryURL: dir, device: device, expecting: .gemma4Toy(),
      integrityPolicy: .trustedReceiptWhenValid)
    #expect(model.integrityPolicy == .fullSha256)
    #expect(throws: (any Error).self) {
      _ = try model.routedExpert(layer: 0, expert: 0)
    }
  }

  @Test func omittedPolicyStaysStrictForLibraryCallers() throws {
    let dir = try Self.writeToySynthetic()
    defer { try? FileManager.default.removeItem(at: dir) }
    try Self.writeVerifiedInstallReceipt(directoryURL: dir)
    let device = try #require(MTLCreateSystemDefaultDevice())

    let model = try Model.load(directoryURL: dir, device: device, expecting: .gemma4Toy())
    #expect(model.integrityPolicy == .fullSha256)
  }
}
