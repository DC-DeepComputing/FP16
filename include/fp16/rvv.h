#pragma once
#ifndef FP16_RVV_H
#define FP16_RVV_H

/* RISC-V RVV 1.0 batch half-precision conversion kernels.
   Compile with -march=rv64gcv_zvl256b_zvfh (or any RVV config with Zvfh).
   On other targets the functions are no-ops guarded by __riscv_vector.

   All kernels are bit-exact with the corresponding scalar fp16.h
   conversions, including NaN payloads, NaN signs, signed zero and
   ARM alternative-format semantics. */

#include <fp16.h>

#if defined(__riscv) && (__riscv_xlen == 64) && defined(__riscv_vector) && defined(__riscv_zvfh)
#include <riscv_vector.h>
#define FP16_HAS_RVV_KERNELS 1

#define FP16_RVV_N 16  /* f32m2 / f16m1 lanes at VLEN=256 */

static inline void fp16_rvv_ieee_to_fp32_bits(
    const uint16_t* input, uint32_t* output, size_t count) {
  size_t i = 0;
  while (i < count) {
    const size_t vl = count - i < FP16_RVV_N ? count - i : FP16_RVV_N;
    const vuint16m1_t hv = __riscv_vle16_v_u16m1(input + i, vl);
    vuint32m2_t f = __riscv_vreinterpret_v_f32m2_u32m2(
        __riscv_vfwcvt_f_f_v_f32m2(__riscv_vreinterpret_v_u16m1_f16m1(hv), vl));
    const vuint16m1_t mag = __riscv_vand_vx_u16m1(hv, 0x7fff, vl);
    /* hardware widens sNaNs to a canonical quiet NaN; the portable
       conversion preserves the payload and the sign — fix those lanes */
    const vbool16_t nan = __riscv_vmand_mm_b16(
        __riscv_vmsgtu_vx_u16m1_b16(mag, 0x7bff, vl),
        __riscv_vmsne_vx_u16m1_b16(mag, 0x7c00, vl), vl);
    const vuint32m2_t fix = __riscv_vor_vv_u32m2(
        __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(
            __riscv_vand_vx_u16m1(hv, 0x8000, vl), vl), 16, vl),
        __riscv_vor_vv_u32m2(__riscv_vmv_v_x_u32m2(0x7f800000u, vl),
            __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(
                __riscv_vand_vx_u16m1(mag, 0x03ff, vl), vl), 13, vl), vl), vl);
    __riscv_vse32_v_u32m2(output + i, __riscv_vmerge_vvm_u32m2(f, fix, nan, vl), vl);
    i += vl;
  }
}

static inline void fp16_rvv_ieee_from_fp32_bits(
    const uint32_t* input, uint16_t* output, size_t count) {
  size_t i = 0;
  while (i < count) {
    const size_t vl = count - i < FP16_RVV_N ? count - i : FP16_RVV_N;
    const vuint32m2_t fv = __riscv_vle32_v_u32m2(input + i, vl);
    const vuint16m1_t h = __riscv_vreinterpret_v_f16m1_u16m1(
        __riscv_vfncvt_f_f_w_f16m1(__riscv_vreinterpret_v_u32m2_f32m2(fv), vl));
    const vuint32m2_t mag = __riscv_vand_vx_u32m2(fv, 0x7fffffffu, vl);
    /* the portable encoder maps every NaN to 0x7e00 with the sign kept */
    const vbool16_t nan = __riscv_vmand_mm_b16(
        __riscv_vmsgtu_vx_u32m2_b16(mag, 0x7f7fffffu, vl),
        __riscv_vmsne_vx_u32m2_b16(mag, 0x7f800000u, vl), vl);
    const vuint16m1_t sgn = __riscv_vnsrl_wx_u16m1(
        __riscv_vsrl_vx_u32m2(__riscv_vand_vx_u32m2(fv, 0x80000000u, vl), 16, vl), 0, vl);
    __riscv_vse16_v_u16m1(output + i,
        __riscv_vmerge_vvm_u16m1(h, __riscv_vor_vx_u16m1(sgn, 0x7e00, vl), nan, vl), vl);
    i += vl;
  }
}

static inline void fp16_rvv_alt_to_fp32_bits(
    const uint16_t* input, uint32_t* output, size_t count) {
  size_t i = 0;
  while (i < count) {
    const size_t vl = count - i < FP16_RVV_N ? count - i : FP16_RVV_N;
    const vuint16m1_t hv = __riscv_vle16_v_u16m1(input + i, vl);
    const vuint16m1_t mag = __riscv_vand_vx_u16m1(hv, 0x7fff, vl);
    /* in ARM alternative format exponent 31 is a regular number
       (65536 + mantissa*128): convert it with an explicit bit pattern */
    const vbool16_t hi = __riscv_vmsgtu_vx_u16m1_b16(mag, 0x7bff, vl);
    const vuint16m1_t adj = __riscv_vmerge_vvm_u16m1(hv,
        __riscv_vor_vv_u16m1(__riscv_vand_vx_u16m1(hv, 0x03ff, vl),
            __riscv_vmv_v_x_u16m1(0x7800, vl), vl), hi, vl);
    const vuint32m2_t f = __riscv_vreinterpret_v_f32m2_u32m2(
        __riscv_vfwcvt_f_f_v_f32m2(__riscv_vreinterpret_v_u16m1_f16m1(adj), vl));
    const vuint32m2_t f2 = __riscv_vor_vv_u32m2(
        __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(
            __riscv_vand_vx_u16m1(hv, 0x8000, vl), vl), 16, vl),
        __riscv_vor_vv_u32m2(__riscv_vmv_v_x_u32m2(0x47800000u, vl),
            __riscv_vsll_vx_u32m2(__riscv_vzext_vf2_u32m2(
                __riscv_vand_vx_u16m1(mag, 0x03ff, vl), vl), 13, vl), vl), vl);
    __riscv_vse32_v_u32m2(output + i, __riscv_vmerge_vvm_u32m2(f, f2, hi, vl), vl);
    i += vl;
  }
}

static inline void fp16_rvv_alt_from_fp32_bits(
    const uint32_t* input, uint16_t* output, size_t count) {
  size_t i = 0;
  while (i < count) {
    const size_t vl = count - i < FP16_RVV_N ? count - i : FP16_RVV_N;
    const vuint32m2_t fv = __riscv_vle32_v_u32m2(input + i, vl);
    const vuint16m1_t h = __riscv_vreinterpret_v_f16m1_u16m1(
        __riscv_vfncvt_f_f_w_f16m1(__riscv_vreinterpret_v_u32m2_f32m2(fv), vl));
    __riscv_vse16_v_u16m1(output + i, h, vl);
    /* alt exp=31 is a non-linear extension above 65536; those lanes are
       rare in practice (none for typical [-1,1] model tensors) and are
       handled by the exact scalar conversion */
    for (size_t k = 0; k < vl; k++) {
      if ((input[i + k] & 0x7fffffffu) >= 0x47800000u) {
        float f;
        __builtin_memcpy(&f, &input[i + k], sizeof(float));
        output[i + k] = fp16_alt_from_fp32_value(f);
      }
    }
    i += vl;
  }
}

#endif  /* RVV guard */
#endif  /* FP16_RVV_H */
