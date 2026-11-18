import Foundation
import Mference
import MferenceEngine

/// Renders what library mode would serve, without loading anything.
///
/// `MferenceServer --library --list-models` and `./mference-ui.sh models` both
/// print this, so the launcher never has its own idea of what is installed:
/// discovery answers, and the script only relays the answer.
public enum ServerLibraryListing {
    /// Printed alone when discovery found nothing. Library mode still starts in
    /// that state — `/v1/models` is simply empty — so this is a normal result
    /// and not an error.
    public static let emptyListing = "no models installed"

    private static let headers = ["MODEL", "FAMILY", "BYTES", "PATH"]

    /// One header row and one row per install, ordered the way `/v1/models`
    /// orders them. `BYTES` is right-aligned; `PATH` is last so a long path
    /// never pushes another column out of alignment.
    public static func text(
        for index: ServerLibraryIndex,
        installedBytes: (ServerLibraryEntry) -> UInt64? =
            ServerLibraryListing.receiptInstalledBytes
    ) -> String {
        guard !index.entries.isEmpty else { return emptyListing }
        var rows = [headers]
        for entry in index.entries {
            rows.append([
                entry.modelID,
                entry.family.rawValue,
                installedBytes(entry).map(String.init) ?? "-",
                entry.directory.path,
            ])
        }
        let widths = (0..<headers.count).map { column in
            rows.map { $0[column].count }.max() ?? 0
        }
        return rows.map { row in
            row.enumerated().map { column, value in
                if column == row.count - 1 { return value }
                let padding = String(repeating: " ", count: widths[column] - value.count)
                // Byte counts read as a column of numbers only when they line
                // up on the right.
                return column == 2 ? padding + value : value + padding
            }.joined(separator: "  ")
        }.joined(separator: "\n")
    }

    /// What library mode serves: one header row and one row per catalog model,
    /// in `/v1/models` order. `BYTES` is the size on disk the catalog recorded
    /// (`-` when unknown).
    public static func text(for catalog: EngineCatalog) -> String {
        guard !catalog.models.isEmpty else { return emptyListing }
        return table(header: ["MODEL", "ENGINE", "BYTES", "PATH"],
                     rows: catalog.models.map {
                         [$0.id, $0.engine.rawValue,
                          $0.sizeBytes > 0 ? String($0.sizeBytes) : "-",
                          $0.location.path]
                     })
    }

    private static func table(header: [String], rows body: [[String]]) -> String {
        let rows = [header] + body
        let widths = (0..<header.count).map { column in
            rows.map { $0[column].count }.max() ?? 0
        }
        return rows.map { row in
            row.enumerated().map { column, value in
                if column == row.count - 1 { return value }
                let padding = String(repeating: " ", count: widths[column] - value.count)
                return column == 2 ? padding + value : value + padding
            }.joined(separator: "  ")
        }.joined(separator: "\n")
    }

    /// Installed size from the verified-install receipt: the sizes it already
    /// records, summed. Metadata only — no directory walk and no weight file is
    /// opened, so listing a 148 GB install costs one small JSON read.
    ///
    /// nil when the receipt cannot be read or records no sizes, which prints as
    /// `-` rather than a misleading `0`.
    public static func receiptInstalledBytes(_ entry: ServerLibraryEntry) -> UInt64? {
        guard let receipt = try? VerifiedInstallReceiptReader.load(
            directoryURL: entry.directory) else { return nil }
        let total = receipt.files.values.reduce(UInt64(0)) { $0 &+ $1.size }
        return total > 0 ? total : nil
    }
}
