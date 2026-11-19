import Foundation
import MferenceLink

/// The phone-only services live in this target, one folder per owner:
/// `Engine`, `Models`, `Memory` (mobile-chat), `Link` (mobile-link),
/// `Scan`, `Calendar`, `Permissions` (mobile-apple). No SwiftUI views here;
/// screens in MferenceMobileUI reach these through protocols so previews and
/// tests run on fakes.
public enum MobileCore {
    /// The link protocol version this build of the phone speaks.
    public static let linkProtocolVersion = LinkProtocol.version
}
