import Foundation
import MferenceMobileUI
import Observation

/// Scan on a sample receipt (Blue Door Cafe), for previews, tests and the
/// skeleton build until mobile-apple's camera and text-recognition model
/// lands. Nothing here opens the camera.
@Observable
public final class SampleScanModel: ScanScreenModel {
    public var scanKind: ScanKind = .receipt
    public private(set) var scanPhase: ScanPhase = .read
    public var scanFields: [ScanField]
    public let scanMerchant = "Blue Door Cafe"

    public init(now: Date = .now) {
        scanFields = [
            ScanField(id: "date", label: "Date", value: now.formatted(date: .abbreviated, time: .omitted), isSure: true),
            ScanField(id: "total", label: "Total", value: "$105.47", isSure: true),
            ScanField(id: "tip", label: "Tip", value: "$16.40", isSure: true),
            ScanField(id: "category", label: "Category", value: "Team meals", isSure: false),
        ]
    }

    /// The shutter: reads the sample receipt after a short pause.
    public func capture() {
        scanPhase = .reading
        Task {
            try? await Task.sleep(for: .milliseconds(900))
            if scanPhase == .reading { scanPhase = .read }
        }
    }

    public func retake() { scanPhase = .viewfinder }

    public func addScanRow() { scanPhase = .added }
}
