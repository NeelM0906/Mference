import Foundation
import Testing
@testable import MferenceRepackCore

@Suite
struct RemoteDownloadSessionTests {
    @Test func typedSessionUsesStallTolerantSerialDefaults() {
        let session = RemoteDownloadSession()
        let configuration = session.configurationSnapshot

        #expect(session.policy.requestTimeoutSeconds == 300)
        #expect(session.policy.resourceTimeoutSeconds == 7 * 24 * 60 * 60)
        #expect(session.policy.waitsForConnectivity)
        #expect(session.policy.maximumConnectionsPerHost == 1)
        #expect(session.policy.maximumRedirects == 5)
        #expect(configuration.requestTimeoutSeconds == 300)
        #expect(configuration.resourceTimeoutSeconds == 7 * 24 * 60 * 60)
        #expect(configuration.waitsForConnectivity)
        #expect(configuration.maximumConnectionsPerHost == 1)
        #expect(configuration.requestCachePolicy == .reloadIgnoringLocalCacheData)
        #expect(!configuration.hasURLCache)
    }

    @Test func injectedPolicyIsAppliedToOwnedConfiguration() {
        let policy = RemoteDownloadSessionPolicy(
            requestTimeoutSeconds: 12,
            resourceTimeoutSeconds: 34,
            waitsForConnectivity: false,
            maximumConnectionsPerHost: 1,
            maximumRedirects: 2)
        let session = RemoteDownloadSession(policy: policy)
        let configuration = session.configurationSnapshot

        #expect(configuration.requestTimeoutSeconds == 12)
        #expect(configuration.resourceTimeoutSeconds == 34)
        #expect(!configuration.waitsForConnectivity)
        #expect(configuration.maximumConnectionsPerHost == 1)
        #expect(session.policy.maximumRedirects == 2)
    }

    /// A 60 s per-request timeout resets only when data arrives, so one stalled
    /// byte range aborts a multi-GB install. Every installer entry point that
    /// omits an explicit session must inherit the stall-tolerant one.
    @Test func installEntryPointsInheritStallTolerantDownloadSession() {
        for options in installEntryPointOptions() {
            #expect(options.downloadSession.policy.requestTimeoutSeconds == 300)
            #expect(options.downloadSession.policy.resourceTimeoutSeconds
                == 7 * 24 * 60 * 60)
        }
    }

    /// Tolerating 5 minutes of silence only works because a genuinely dead
    /// connection is still bounded by the retry policy.
    @Test func installEntryPointsKeepBoundedRetries() {
        for options in installEntryPointOptions() {
            let policy = RemoteRetryPolicy(attempts: options.rangeRetryAttempts,
                                           baseDelayNs: options.retryBaseDelayNs)
            #expect(policy.attempts == 4)
            #expect(policy.baseDelayNs == 1_000_000_000)
            #expect(policy.maxDelayNs == 16_000_000_000)
        }
    }

    /// Options as the CLI, the Mac app installer client and every pinned source
    /// build them: no explicit session, no explicit retry tuning.
    private func installEntryPointOptions() -> [RemoteStreamingRepackOptions] {
        let outputDirectory = URL(fileURLWithPath: "/tmp/mference-install")
        return [RemoteStreamingRepackOptions(repoID: "owner/model",
                                             revision: "main",
                                             outputDir: outputDirectory.path)]
            + SupportedModelSource.all.map {
                $0.installOptions(outputDirectory: outputDirectory,
                                  overwrite: true,
                                  token: nil)
            }
    }

    @Test func metadataRedirectsFollowOnlyBoundedSameHostHTTPS() {
        var original = URLRequest(url: URL(string: "https://hf.test/model/file")!)
        original.httpMethod = "HEAD"
        original.setValue("Bearer secret", forHTTPHeaderField: "Authorization")
        var policy = RemoteMetadataRedirectPolicy(
            originalRequest: original,
            maximumRedirects: 2)

        let sameHost = policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://hf.test/api/cache/file?etag=value")!))
        #expect(sameHost?.httpMethod == "HEAD")
        #expect(sameHost?.value(forHTTPHeaderField: "Accept-Encoding") == "identity")
        #expect(sameHost?.value(forHTTPHeaderField: "Authorization") == "Bearer secret")

        #expect(policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://storage.test/signed?token=private")!)) == nil)
        #expect(policy.request(proposedRequest: URLRequest(
            url: URL(string: "https://hf.test/too-many")!)) == nil)
    }
}
