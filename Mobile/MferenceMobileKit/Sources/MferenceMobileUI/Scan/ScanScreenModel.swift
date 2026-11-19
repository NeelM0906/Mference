import Foundation
import Observation

/// What the Scan screens read and do (camera, reading, the filled fields and
/// adding the row). The screens depend on this protocol only.
///
/// Owned by mobile-apple: the live model drives the camera, Apple's text
/// recognition and the field filling. `SampleScanModel`
/// (MferenceMobileSamples) stands in for previews, tests and the skeleton
/// build.
public protocol ScanScreenModel: AnyObject, Observable {
    var scanKind: ScanKind { get set }
    /// The shell reads it for the dark ground on Duo open.
    var scanPhase: ScanPhase { get }
    var scanFields: [ScanField] { get set }
    var scanMerchant: String { get }

    /// The shutter.
    func capture()
    func retake()
    func addScanRow()
}
