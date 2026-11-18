import Foundation
import FoundationModels

/// Apple's on-device model (FoundationModels), decision L12. No permission is
/// asked; when Apple Intelligence is off or not ready, `availability()` says
/// so and `AppleModelService` uses Quick instead.
public struct FoundationTextModel: AssistantTextModel {
    public init() {}

    public var kind: AssistantModelKind { .appleOnDevice }

    public func availability() async -> AssistantAvailability {
        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                return .unavailable(reason: "This Mac cannot run Apple Intelligence.")
            case .appleIntelligenceNotEnabled:
                return .unavailable(reason: "Apple Intelligence is turned off in System Settings.")
            case .modelNotReady:
                return .unavailable(reason: "Apple's on-device model is still getting ready.")
            @unknown default:
                return .unavailable(reason: "Apple's on-device model is not available.")
            }
        }
    }

    public func complete(_ request: AssistantRequest) async throws -> String {
        let session = LanguageModelSession(instructions: request.instructions)
        let options = FoundationModels.GenerationOptions(temperature: request.temperature,
                                                         maximumResponseTokens: request.maximumTokens)
        do {
            return try await session.respond(to: request.prompt, options: options).content
        } catch let error as LanguageModelSession.GenerationError {
            throw Self.map(error)
        }
    }

    static func map(_ error: LanguageModelSession.GenerationError) -> AssistantModelError {
        switch error {
        case .exceededContextWindowSize:
            .inputTooLong
        case .assetsUnavailable:
            .noModelAvailable(reason: "Apple's on-device model is not available right now.")
        case .guardrailViolation, .refusal:
            .declined("Apple's on-device model declined this text.")
        case .unsupportedLanguageOrLocale:
            .declined("Apple's on-device model does not support this language.")
        case .decodingFailure, .unsupportedGuide:
            .unreadableAnswer
        case .rateLimited, .concurrentRequests:
            .noModelAvailable(reason: "Apple's on-device model is busy. Try again in a moment.")
        @unknown default:
            .noModelAvailable(reason: "Apple's on-device model is not available right now.")
        }
    }
}
