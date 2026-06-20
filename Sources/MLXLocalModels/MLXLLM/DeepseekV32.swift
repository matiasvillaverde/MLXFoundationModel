// Copyright (c) 2026 Matias Villaverde

import MLX

internal typealias DeepseekV32Configuration = GLM4MoELiteConfiguration

/// DeepSeek V3.2 uses the same DSA attention layout as GLM MoE DSA in mlx-lm.
///
/// Keep a named adapter type so DeepSeek-specific optimizations such as
/// IndexCache and FP8 scale-inverse loading can be added without coupling them
/// to GLM's checkpoint-defined full/shared indexer schedule.
internal class DeepseekV32Model: GLM4MoELiteModel {
    internal override init(_ args: DeepseekV32Configuration) {
        super.init(args)
    }

    internal override func newCache(parameters: GenerateParameters?) -> [KVCache] {
        guard let frequency = parameters?.indexCacheFrequency, frequency >= 2 else {
            return super.newCache(parameters: parameters)
        }

        return configuration.dsaIndexerKinds.indices.map { layerIndex in
            if layerIndex % frequency == 0 {
                return CacheList(
                    Self.makeBaseCache(parameters: parameters),
                    Self.makeBaseCache(parameters: parameters)
                )
            }
            return CacheList(Self.makeBaseCache(parameters: parameters))
        }
    }

    private static func makeBaseCache(parameters: GenerateParameters?) -> KVCache {
        if let maxKVSize = parameters?.maxKVSize {
            return RotatingKVCache(
                maxSize: maxKVSize,
                keep: GenerationConstants.rotatingCacheKeepTokens
            )
        }
        return KVCacheSimple()
    }
}
