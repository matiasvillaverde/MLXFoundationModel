// Copyright © 2025 Apple Inc.

import Foundation
import MLX

/// Base ``LanguageModel`` configuration -- provides `modelType`
/// and `quantization` (used in loading the model).
///
/// This is used by ``ModelFactory/load(hub:configuration:progressHandler:)``
/// to determine the type of model to load.
internal struct BaseConfiguration: Codable, Sendable {
    internal let modelType: String
    internal var eosTokenIds: IntOrIntArray?

    internal struct Quantization: Codable, Sendable, Equatable {
        internal init(groupSize: Int, bits: Int, quantizationMode: String? = nil) {
            self.groupSize = groupSize
            self.bits = bits
            self.quantizationMode = quantizationMode
        }

        internal let groupSize: Int
        internal let bits: Int
        internal var quantMethod: String? = nil
        internal var linearClass: String? = nil
        internal var quantizationMode: String? = nil

        internal var mode: QuantizationMode {
            if let quantizationMode, let mode = QuantizationMode(rawValue: quantizationMode) {
                return mode
            }
            return .affine
        }

        internal var asTuple: (groupSize: Int, bits: Int, mode: QuantizationMode) {
            (groupSize, bits, mode)
        }

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits = "bits"
            case quantMethod = "quant_method"
            case linearClass = "linear_class"
            case quantizationMode = "quantization_mode"
            case mode = "mode"
        }

        init(from decoder: any Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            groupSize = try container.decode(Int.self, forKey: .groupSize)
            bits = try container.decode(Int.self, forKey: .bits)
            quantMethod = try container.decodeIfPresent(String.self, forKey: .quantMethod)
            linearClass = try container.decodeIfPresent(String.self, forKey: .linearClass)
            quantizationMode =
                try container.decodeIfPresent(String.self, forKey: .quantizationMode)
                ?? container.decodeIfPresent(String.self, forKey: .mode)
        }

        func encode(to encoder: any Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(groupSize, forKey: .groupSize)
            try container.encode(bits, forKey: .bits)
            try container.encodeIfPresent(quantMethod, forKey: .quantMethod)
            try container.encodeIfPresent(linearClass, forKey: .linearClass)
            try container.encodeIfPresent(quantizationMode, forKey: .quantizationMode)
        }
    }

    /// handling instructions for ``PerLayerQuantization``
    internal enum QuantizationOption: Sendable {
        case skip
        case quantize(Quantization)
    }

    /// Per-layer ``Quantization`` values with optional default.
    internal struct PerLayerQuantization: Sendable {
        internal var quantization: Quantization? = nil
        internal var perLayerQuantization: [String: QuantizationOption]

        internal init(
            quantization: BaseConfiguration.Quantization? = nil,
            perLayerQuantization: [String: BaseConfiguration.QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        /// The quantization to apply for the given layer name or nil for no quantization.
        internal func quantization(layer: String) -> Quantization? {
            if let perLayer = perLayerQuantization[layer] {
                switch perLayer {
                case .skip:
                    return nil
                case .quantize(let quantization):
                    return quantization
                }
            } else {
                return quantization
            }
        }
    }

    /// Special codable to support a mixed key: Int / key: Quantization
    /// structure for hereogenous quantization, e.g.
    ///
    /// ```
    /// "quantization": {
    ///     "group_size": 64,
    ///     "bits": 4,
    ///     "model.embed_tokens": {
    ///         "group_size": 32,
    ///         "bits": 4
    ///     },
    ///     "model.layers.0.self_attn.q_norm": false,
    /// ```
    ///
    /// This mixed type structure requires manual decoding.
    struct QuantizationContainer: Codable, Sendable {
        var quantization: Quantization
        var perLayerQuantization: PerLayerQuantization

        // based on Dictionary's coding key
        internal struct _DictionaryCodingKey: CodingKey {
            internal let stringValue: String
            internal let intValue: Int?

            internal init(stringValue: String) {
                self.stringValue = stringValue
                self.intValue = Int(stringValue)
            }

            internal init(intValue: Int) {
                self.stringValue = "\(intValue)"
                self.intValue = intValue
            }
        }

        init(from decoder: any Decoder) throws {
            // handle the embedded Quantization
            self.quantization = try Quantization(from: decoder)

            // and the interleaved per-layer values
            var perLayerQuantization = [String: QuantizationOption]()
            let container = try decoder.container(keyedBy: _DictionaryCodingKey.self)
            for key in container.allKeys {
                switch key.stringValue {
                case Quantization.CodingKeys.groupSize.rawValue: continue
                case Quantization.CodingKeys.bits.rawValue: continue
                case Quantization.CodingKeys.quantMethod.rawValue: continue
                case Quantization.CodingKeys.linearClass.rawValue: continue
                case Quantization.CodingKeys.quantizationMode.rawValue: continue
                case Quantization.CodingKeys.mode.rawValue: continue

                default:
                    if let f = try? container.decode(Bool.self, forKey: key) {
                        if !f {
                            perLayerQuantization[key.stringValue] = .skip
                        }
                    } else {
                        perLayerQuantization[key.stringValue] = .quantize(
                            try container.decode(Quantization.self, forKey: key))
                    }
                }
            }
            self.perLayerQuantization = PerLayerQuantization(
                quantization: quantization, perLayerQuantization: perLayerQuantization)
        }

        func encode(to encoder: any Encoder) throws {
            try quantization.encode(to: encoder)

            var container = encoder.container(keyedBy: _DictionaryCodingKey.self)
            for (key, value) in perLayerQuantization.perLayerQuantization {
                switch value {
                case .skip:
                    try container.encode(false, forKey: .init(stringValue: key))
                case .quantize(let q):
                    try container.encode(q, forKey: .init(stringValue: key))
                }
            }
        }
    }

    var quantizationContainer: QuantizationContainer?

    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    internal var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    internal var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case eosTokenIds = "eos_token_id"
        case quantizationContainer = "quantization"
    }
}
