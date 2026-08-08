import Metal

public struct ForwardRuntime: Sendable {
    public let producer: any ContinuableLogitProducer
    public let prefillConfig: PrefillRuntimeConfig

    init(producer: any ContinuableLogitProducer, prefillConfig: PrefillRuntimeConfig) {
        self.producer = producer
        self.prefillConfig = prefillConfig
    }
}

public enum ForwardRunnerFactory {
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> ForwardRuntime {
        if model.config.family == .maple {
            return ForwardRuntime(producer: try MapleForwardRunner(
                model: model, context: context, maxContext: maxContext), prefillConfig: .off)
        }
        return ForwardRuntime(producer: try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration),
            prefillConfig: runtimeConfiguration.prefillConfig)
    }
}
