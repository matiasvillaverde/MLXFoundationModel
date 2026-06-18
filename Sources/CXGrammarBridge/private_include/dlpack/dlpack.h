/*
 * Minimal DLPack C ABI declarations needed by XGrammar's public matcher API.
 * Compatible with DLPack 1.0 tensor/device/data-type layouts.
 */
#ifndef DLPACK_DLPACK_H_
#define DLPACK_DLPACK_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
typedef enum : int32_t {
#else
typedef enum {
#endif
  kDLCPU = 1,
  kDLCUDA = 2,
  kDLCUDAHost = 3,
  kDLOpenCL = 4,
  kDLVulkan = 7,
  kDLMetal = 8,
  kDLVPI = 9,
  kDLROCM = 10,
  kDLROCMHost = 11,
  kDLExtDev = 12,
  kDLCUDAManaged = 13,
  kDLOneAPI = 14,
  kDLWebGPU = 15,
  kDLHexagon = 16,
  kDLMAIA = 17,
} DLDeviceType;

typedef struct {
  DLDeviceType device_type;
  int32_t device_id;
} DLDevice;

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
  kDLBool = 6U,
} DLDataTypeCode;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct {
  void *data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} DLTensor;

#ifdef __cplusplus
}
#endif

#endif
