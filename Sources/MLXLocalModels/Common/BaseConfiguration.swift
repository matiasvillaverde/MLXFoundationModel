import Foundation
import MLX

/// Minimal model metadata needed before the concrete architecture type is known.
internal struct BaseConfiguration: Codable, Sendable {
    internal let modelType: String
    internal var eosTokenIds: IntOrIntArray?
    internal var quantizationContainer: QuantizationContainer?

    internal struct Quantization: Codable, Sendable, Equatable {
        internal let groupSize: Int
        internal let bits: Int
        internal var quantMethod: String?
        internal var linearClass: String?
        internal var quantizationMode: String?

        internal init(groupSize: Int, bits: Int) {
            self.groupSize = groupSize
            self.bits = bits
        }

        internal init(groupSize: Int, bits: Int, quantizationMode: String?) {
            self.groupSize = groupSize
            self.bits = bits
            self.quantizationMode = quantizationMode
        }

        internal var mode: QuantizationMode {
            quantizationMode.flatMap(QuantizationMode.init(rawValue:)) ?? .affine
        }

        internal var asTuple: (groupSize: Int, bits: Int, mode: QuantizationMode) {
            (groupSize, bits, mode)
        }

        internal enum CodingKeys: String, CodingKey, CaseIterable {
            case groupSize = "group_size"
            case bits
            case quantMethod = "quant_method"
            case linearClass = "linear_class"
            case quantizationMode = "quantization_mode"
            case mode
        }

        internal init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            groupSize = try container.decode(Int.self, forKey: .groupSize)
            bits = try container.decode(Int.self, forKey: .bits)
            quantMethod = try container.decodeIfPresent(String.self, forKey: .quantMethod)
            linearClass = try container.decodeIfPresent(String.self, forKey: .linearClass)
            quantizationMode =
                try container.decodeIfPresent(String.self, forKey: .quantizationMode)
                ?? container.decodeIfPresent(String.self, forKey: .mode)
        }

        internal func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)

            try container.encode(groupSize, forKey: .groupSize)
            try container.encode(bits, forKey: .bits)
            try container.encodeIfPresent(quantMethod, forKey: .quantMethod)
            try container.encodeIfPresent(linearClass, forKey: .linearClass)
            try container.encodeIfPresent(quantizationMode, forKey: .quantizationMode)
        }
    }

    internal enum QuantizationOption: Sendable, Equatable {
        case skip
        case quantize(Quantization)
    }

    internal struct PerLayerQuantization: Sendable, Equatable {
        internal var quantization: Quantization?
        internal var perLayerQuantization: [String: QuantizationOption]

        internal init(
            quantization: Quantization?,
            perLayerQuantization: [String: QuantizationOption]
        ) {
            self.quantization = quantization
            self.perLayerQuantization = perLayerQuantization
        }

        internal func quantization(layer: String) -> Quantization? {
            guard let override = perLayerQuantization[layer] else {
                return quantization
            }

            switch override {
            case .skip:
                return nil
            case .quantize(let quantization):
                return quantization
            }
        }
    }

    internal struct QuantizationContainer: Codable, Sendable, Equatable {
        internal var quantization: Quantization
        internal var perLayerQuantization: PerLayerQuantization

        internal init(from decoder: Decoder) throws {
            quantization = try Quantization(from: decoder)
            perLayerQuantization = try Self.decodePerLayerQuantization(
                from: decoder,
                defaultQuantization: quantization
            )
        }

        internal func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: DynamicCodingKey.self)

            try container.encode(quantization.groupSize, forKey: .key("group_size"))
            try container.encode(quantization.bits, forKey: .key("bits"))
            try container.encodeIfPresent(quantization.quantMethod, forKey: .key("quant_method"))
            try container.encodeIfPresent(quantization.linearClass, forKey: .key("linear_class"))
            try container.encodeIfPresent(
                quantization.quantizationMode,
                forKey: .key("quantization_mode")
            )

            for (layer, option) in perLayerQuantization.perLayerQuantization {
                switch option {
                case .skip:
                    try container.encode(false, forKey: .key(layer))
                case .quantize(let quantization):
                    try container.encode(quantization, forKey: .key(layer))
                }
            }
        }

        private static func decodePerLayerQuantization(
            from decoder: Decoder,
            defaultQuantization: Quantization
        ) throws -> PerLayerQuantization {
            let container = try decoder.container(keyedBy: DynamicCodingKey.self)
            var overrides = [String: QuantizationOption]()

            for key in container.allKeys where !Self.reservedQuantizationKeys.contains(key.stringValue) {
                if let bool = try? container.decode(Bool.self, forKey: key) {
                    if bool == false {
                        overrides[key.stringValue] = .skip
                    }
                } else {
                    overrides[key.stringValue] = .quantize(
                        try container.decode(Quantization.self, forKey: key)
                    )
                }
            }

            return PerLayerQuantization(
                quantization: defaultQuantization,
                perLayerQuantization: overrides
            )
        }

        private static let reservedQuantizationKeys = Set(
            Quantization.CodingKeys.allCases.map(\.rawValue)
        )
    }

    @available(*, deprecated, message: "Please use perLayerQuantization instead")
    internal var quantization: Quantization? {
        quantizationContainer?.quantization
    }

    internal var perLayerQuantization: PerLayerQuantization? {
        quantizationContainer?.perLayerQuantization
    }

    internal enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case eosTokenIds = "eos_token_id"
        case quantizationContainer = "quantization"
    }
}

private struct DynamicCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init(stringValue: String) {
        self.stringValue = stringValue
        self.intValue = Int(stringValue)
    }

    init(intValue: Int) {
        self.stringValue = String(intValue)
        self.intValue = intValue
    }

    static func key(_ value: String) -> Self {
        Self(stringValue: value)
    }
}
