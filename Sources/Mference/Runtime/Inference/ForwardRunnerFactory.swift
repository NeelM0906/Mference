import Metal

public struct ForwardRuntime: Sendable {
    public let producer: any ContinuableLogitProducer
    public let prefillConfig: PrefillRuntimeConfig
    /// Construction cannot know the executed path. Read the per-request report.
    @available(*, deprecated, message: "Use RawDecodeResult.prefillExecution after generation")
    public var executedPrefillMode: PrefillExecutedMode { .unreported }
    public let kvStorageMode: PrefillKVStorageMode

    init(producer: any ContinuableLogitProducer,
         prefillConfig: PrefillRuntimeConfig,
         kvStorageMode: PrefillKVStorageMode) {
        self.producer = producer
        self.prefillConfig = prefillConfig
        self.kvStorageMode = kvStorageMode
    }
}

public enum ForwardRunnerFactory {
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> ForwardRuntime {
        let maximum = model.config.family.maximumContext
        guard maxContext <= maximum else {
            throw ContextLimitError(family: model.config.family, requested: maxContext, maximum: maximum)
        }
        if model.config.family == .maple {
            return ForwardRuntime(producer: try MapleForwardRunner(
                model: model, context: context, maxContext: maxContext,
                useFlashHead: runtimeConfiguration.useMapleFlashHead),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  kvStorageMode: .bf16)
        }
        if model.config.family == .qwen38 {
            return ForwardRuntime(producer: try Qwen38ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .minicpm5 {
            return ForwardRuntime(producer: try MiniCPM5ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .glm53Flash {
            return ForwardRuntime(producer: try Glm53ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .qwen38flashnext {
            return ForwardRuntime(producer: try FlashNextForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  kvStorageMode: .fp16)
        }
        return ForwardRuntime(producer: try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration),
            prefillConfig: runtimeConfiguration.prefillConfig,
            kvStorageMode: .fp16)
    }
}
