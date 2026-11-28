//===- runtime_support.h - C the generated CPU code may call ----*- C++ -*-===//
#ifndef BACKEND_CPU_CODEGEN_RUNTIME_SUPPORT_H
#define BACKEND_CPU_CODEGEN_RUNTIME_SUPPORT_H

namespace backend {

// memref.copy between non-contiguous memrefs lowers to a call of this
// function (normally MLIR's c_runner_utils). Freestanding, so the same source
// compiles for every target without a sysroot. toolchain.cpp writes it to
// <dir>/runtime_support.c and compiles it with clang.
constexpr const char *kRuntimeSupport = R"c(
typedef __INTPTR_TYPE__ intptr_t;
typedef __SIZE_TYPE__ size_t;
void *memcpy(void *dst, const void *src, size_t n);
typedef struct { intptr_t rank; void *descriptor; } unranked_memref;

/* Descriptor layout: { allocated, aligned, offset, sizes[rank], strides[rank] }.
   Ranks above 8 are not supported: such a copy is silently skipped. */
void memrefCopy(intptr_t elem_size, unranked_memref *src, unranked_memref *dst) {
  intptr_t rank = src->rank;
  char *s_base = ((char **)src->descriptor)[1];
  char *d_base = ((char **)dst->descriptor)[1];
  intptr_t *s_meta = (intptr_t *)((char **)src->descriptor + 2);
  intptr_t *d_meta = (intptr_t *)((char **)dst->descriptor + 2);
  intptr_t s_off = s_meta[0], d_off = d_meta[0];
  intptr_t *sizes = s_meta + 1, *s_strides = s_meta + 1 + rank, *d_strides = d_meta + 1 + rank;
  if (rank == 0) {
    memcpy(d_base + d_off * elem_size, s_base + s_off * elem_size, (size_t)elem_size);
    return;
  }
  for (intptr_t i = 0; i < rank; ++i)
    if (sizes[i] == 0) return;
  intptr_t index[8] = {0};
  if (rank > 8) return;
  int contiguous = s_strides[rank - 1] == 1 && d_strides[rank - 1] == 1;
  intptr_t run = contiguous ? sizes[rank - 1] : 1;
  for (;;) {
    intptr_t so = s_off, d = d_off;
    for (intptr_t i = 0; i < rank; ++i) {
      so += index[i] * s_strides[i];
      d += index[i] * d_strides[i];
    }
    memcpy(d_base + d * elem_size, s_base + so * elem_size, (size_t)(run * elem_size));
    intptr_t axis = contiguous ? rank - 2 : rank - 1;
    for (; axis >= 0; --axis) {
      if (++index[axis] < sizes[axis]) break;
      index[axis] = 0;
    }
    if (axis < 0) return;
  }
}
)c";

} // namespace backend

#endif // BACKEND_CPU_CODEGEN_RUNTIME_SUPPORT_H
