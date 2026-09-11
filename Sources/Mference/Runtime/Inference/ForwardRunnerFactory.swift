import Metal

public struct ForwardRuntime: Sendable {
    public let producer: any ContinuableLogitProducer
    public let prefillConfig: PrefillRuntimeConfig
    public let executedPrefillMode: PrefillExecutedMode
    public let kvStorageMode: PrefillKVStorageMode

    init(producer: any ContinuableLogitProducer,
         prefillConfig: PrefillRuntimeConfig,
         executedPrefillMode: PrefillExecutedMode,
         kvStorageMode: PrefillKVStorageMode) {
        self.producer = producer
        self.prefillConfig = prefillConfig
        self.executedPrefillMode = executedPrefillMode
        self.kvStorageMode = kvStorageMode
    }
}

public enum ForwardRunnerFactory {
    public static func make(model: Model,
                            context: MetalContext,
                            maxContext: Int,
                            runtimeConfiguration: RuntimeConfiguration = .production) throws -> ForwardRuntime {
        if model.config.family == .maple {
            return ForwardRuntime(producer: try MapleForwardRunner(
                model: model, context: context, maxContext: maxContext,
                useFlashHead: runtimeConfiguration.useMapleFlashHead),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                                      ? .chunked : .off,
                                  kvStorageMode: .bf16)
        }
        if model.config.family == .qwen38 {
            return ForwardRuntime(producer: try Qwen38ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                                      ? .chunked : .off,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .minicpm5 {
            return ForwardRuntime(producer: try MiniCPM5ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                                      ? .chunked : .off,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .glm53Flash {
            // Per-token runner; `prefillChunked` walks the chunk through the
            // decode path, so the chunked config is honored by construction.
            return ForwardRuntime(producer: try Glm53ForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                                      ? .chunked : .off,
                                  kvStorageMode: .fp16)
        }
        if model.config.family == .qwen38flashnext {
            return ForwardRuntime(producer: try FlashNextForwardRunner(
                model: model, context: context, maxContext: maxContext,
                runtimeConfiguration: runtimeConfiguration),
                                  prefillConfig: runtimeConfiguration.prefillConfig,
                                  executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                                      ? .chunked : .off,
                                  kvStorageMode: .fp16)
        }
        return ForwardRuntime(producer: try RealForwardRunner(
            model: model,
            context: context,
            maxContext: maxContext,
            runtimeConfiguration: runtimeConfiguration),
            prefillConfig: runtimeConfiguration.prefillConfig,
            executedPrefillMode: runtimeConfiguration.prefillConfig.mode == .chunked
                ? .chunked : .off,
            kvStorageMode: .fp16)
    }
}
