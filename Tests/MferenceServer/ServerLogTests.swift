import Testing
@testable import MferenceServerCore

@Suite struct ServerLogTests {
    @Test(arguments: [false, true])
    func cancellationIsNotReportedAsServerFailure(streaming: Bool) {
        let line = ServerLog.requestFailureMessage(id: "test-request", status: 500,
            streaming: streaming, error: CancellationError())
        #expect(line == "request test-request cancelled streaming=\(streaming)")
        #expect(!line.contains("failed") && !line.contains("500"))
    }

    @Test func actualFailuresRetainStatusAndDetails() {
        struct DecodeFailure: Error {}
        let line = ServerLog.requestFailureMessage(id: "test-request", status: 500,
            streaming: true, error: DecodeFailure())
        #expect(line == "request test-request failed status=500 streaming=true error=DecodeFailure()")
        let request = ServerLog.requestFailureMessage(id: "test-request", status: 400,
            streaming: false, error: ServerRequestError.invalid(message: "test detail", param: "messages", code: "bad_test"))
        #expect(request == "request test-request failed status=400 streaming=false error=bad_test: test detail")
    }
}
