import Testing
import Foundation
@testable import Mference

/// Rot check for `docs/FAMILY_CONTRACT.md`.
///
/// `ArchConfig` is the declarative family contract and the document is its
/// canonical enumeration, so the two drift apart the moment an axis is added
/// without a row. Reflection over every registered family baseline — and over
/// the nested config structs those baselines hold — recovers the full set of
/// stored-property names, and each name must appear in the document as an
/// inline-code span. A new axis therefore fails this suite until it is
/// documented, and the failure names the fields that are missing.
@Suite struct FamilyContractDocTests {

    /// `docs/FAMILY_CONTRACT.md`, located from this source file rather than
    /// from the working directory or an environment variable so the check
    /// cannot be defeated by where the suite is run from.
    private static var contractURL: URL {
        var root = URL(fileURLWithPath: #filePath)
        // Tests/Mference/Core/Infrastructure/ModelIO/<this file> -> repo root.
        for _ in 0..<6 { root.deleteLastPathComponent() }
        return root.appendingPathComponent("docs/FAMILY_CONTRACT.md")
    }

    /// Stored-property labels of `value`, descending into nested config
    /// structs such as `LinearAttentionConfig`.
    ///
    /// Only labelled struct children are followed, which leaves `[UInt8]`
    /// masks (collection display style), enums, strings, and the numeric
    /// scalars as leaves.
    private static func axisLabels(of value: Any) -> Set<String> {
        var labels: Set<String> = []
        for child in Mirror(reflecting: value).children {
            guard let label = child.label else { continue }
            labels.insert(label)
            let nested = Mirror(reflecting: child.value)
            guard nested.displayStyle == .struct,
                  !nested.children.isEmpty,
                  nested.children.allSatisfy({ $0.label != nil })
            else { continue }
            labels.formUnion(axisLabels(of: child.value))
        }
        return labels
    }

    /// Union of the axis names across every family in `knownArchitectures`,
    /// so an axis that only one family populates is still enumerated.
    private static var allAxisLabels: Set<String> {
        ArchConfig.knownArchitectures.values.reduce(into: Set<String>()) {
            $0.formUnion(axisLabels(of: $1))
        }
    }

    @Test func contractDocumentsEveryArchConfigAxis() throws {
        let url = Self.contractURL
        #expect(FileManager.default.fileExists(atPath: url.path),
                "missing \(url.path)")
        let doc = try String(contentsOf: url, encoding: .utf8)

        let missing = Self.allAxisLabels
            .filter { !doc.contains("`\($0)`") }
            .sorted()
        #expect(missing.isEmpty, """
            docs/FAMILY_CONTRACT.md does not document \(missing.count) \
            ArchConfig axis/axes as `name`: \(missing.joined(separator: ", ")). \
            Add a row per the document's "How to add an axis" section.
            """)
    }

    /// Guards the reflection itself. If `axisLabels` stopped descending into
    /// the nested config structs, the check above would pass vacuously for
    /// every axis those structs own, so probe one field from each of the four.
    @Test func reflectionReachesNestedConfigAxes() {
        let labels = Self.allAxisLabels
        let probes = [
            "hiddenSize",           // ArchConfig itself
            "convKernelSize",       // LinearAttentionConfig
            "ropeScalingBetaSlow",  // CompressedAttentionConfig
            "sinkhornIters",        // HyperConnectionConfig
            "logScalingAlpha",      // RelativePositionConfig
        ]
        for probe in probes {
            #expect(labels.contains(probe),
                    "reflection over ArchConfig did not reach \(probe)")
        }
    }

    /// The document's family legend has to name every registered family, or
    /// the "observed values" columns are quietly incomplete.
    @Test func contractNamesEveryRegisteredFamily() throws {
        let doc = try String(contentsOf: Self.contractURL, encoding: .utf8)
        for family in ArchConfig.knownArchitectures.keys.sorted(by: {
            $0.rawValue < $1.rawValue
        }) {
            #expect(doc.contains("`\(family.rawValue)`"),
                    "docs/FAMILY_CONTRACT.md does not name family `\(family.rawValue)`")
        }
    }
}
