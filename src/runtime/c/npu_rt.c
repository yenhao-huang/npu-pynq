/* NPU runtime linked into every compiled model.
 *
 * Generated code calls npu_rt_gemm(program id, A, partials) for each NPU
 * task. The runtime packs the INT8 left operand into A tiles in the I/O
 * arena (src/isa/layout.py), runs the task's ISA program, and unpacks the
 * INT32 partial planes. Two backends execute programs:
 *
 *   NPU_RT_PYNQ  the FPGA: PROG_ADDR/PROG_LEN/DATA_BASE, START, poll STATUS
 *   NPU_RT_SIM   a C port of src/isa/sim.py over host memory, for the Mac
 *
 * The code is freestanding (no libc headers) so the same file cross-compiles
 * for armv7a-linux-gnueabihf without a sysroot; memcpy/memset resolve at
 * load time against the host process.
 */

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef __INTPTR_TYPE__ intptr_t;
typedef __SIZE_TYPE__ size_t;

void *memcpy(void *dst, const void *src, size_t n);
void *memset(void *dst, int c, size_t n);

enum { NPU_RT_SIM = 0, NPU_RT_PYNQ = 1 };

/* One program record per NPU task, filled by the loader from programs.json. */
typedef struct {
  int32_t word_offset, length;
  int32_t m, n, k, tm, tn, tk;
  int32_t a_bytes, c_base, c_bytes, b_bytes;
  /* Weight streaming: when stage_src is set the task's weight block is copied
   * from it into the stage window before the program runs. */
  const uint8_t *stage_src;
} npu_program;

typedef struct {
  uint32_t phys;
  uint8_t *host;
  uint32_t size;
} npu_region;

typedef struct {
  int32_t backend;
  volatile uint32_t *mmio;
  uint8_t *io;
  uint32_t io_phys, io_size;
  const uint64_t *words; /* program words as the NPU sees them (relocated) */
  uint32_t words_phys;
  uint8_t *stage;
  const npu_program *programs;
  int32_t count;
  npu_region regions[4];
  int32_t nregions;
  /* Telemetry, read back by the engine. */
  int64_t calls, jobs, cycles, macs, polls;
  int32_t error, error_program;
} npu_rt_state;

static npu_rt_state rt;

/* MLIR C-interface descriptors for memref<?xi8> and memref<?xi32>. */
typedef struct {
  int8_t *allocated, *aligned;
  intptr_t offset, size, stride;
} memref_i8;
typedef struct {
  int32_t *allocated, *aligned;
  intptr_t offset, size, stride;
} memref_i32;

#define REG_ISA_CONTROL 0x40
#define REG_ISA_STATUS 0x44
#define REG_PROG_ADDR 0x48
#define REG_PROG_LEN 0x4C
#define REG_DATA_BASE 0x50
#define REG_ISA_ERROR 0x54
#define REG_ISA_JOBS 0x5C
#define REG_ISA_CYCLES 0x60

static inline void barrier(void) {
#if defined(__arm__)
  __asm__ volatile("dsb sy" ::: "memory");
#else
  __asm__ volatile("" ::: "memory");
#endif
}

static int32_t align8(int32_t v) { return (v + 7) & ~7; }
static int32_t min32(int32_t a, int32_t b) { return a < b ? a : b; }
static int32_t cdiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

/* --- layout (port of src/isa/layout.py) -------------------------------- */
static int32_t m_i(const npu_program *p, int32_t mi) { return min32(p->tm, p->m - mi * p->tm); }
static int32_t n_i(const npu_program *p, int32_t ni) { return min32(p->tn, p->n - ni * p->tn); }
static int32_t k_c(const npu_program *p, int32_t kc) { return min32(p->tk, p->k - kc * p->tk); }
static int32_t a_offset(const npu_program *p, int32_t mi, int32_t kc) {
  return mi * cdiv(p->k, p->tk) * align8(p->tm * p->tk) + kc * align8(m_i(p, mi) * p->tk);
}
static int32_t c_tile_stride(const npu_program *p, int32_t mi) { return align8(4 * m_i(p, mi) * p->tn); }

static void pack_a(const npu_program *p, const int8_t *a, uint8_t *dst) {
  int32_t mt = cdiv(p->m, p->tm), kt = cdiv(p->k, p->tk);
  for (int32_t mi = 0; mi < mt; ++mi) {
    int32_t rows = m_i(p, mi);
    for (int32_t kc = 0; kc < kt; ++kc) {
      int32_t cols = k_c(p, kc);
      uint8_t *tile = dst + a_offset(p, mi, kc);
      const int8_t *src = a + (int64_t)mi * p->tm * p->k + kc * p->tk;
      for (int32_t r = 0; r < rows; ++r)
        memcpy(tile + r * cols, src + (int64_t)r * p->k, (size_t)cols);
    }
  }
}

static void unpack_c(const npu_program *p, const uint8_t *src, int32_t *c) {
  int32_t mt = cdiv(p->m, p->tm), nt = cdiv(p->n, p->tn), kt = cdiv(p->k, p->tk);
  int32_t plane = 0;
  for (int32_t mi = 0; mi < mt; ++mi)
    plane += nt * c_tile_stride(p, mi);
  for (int32_t kc = 0; kc < kt; ++kc) {
    int32_t row_base = 0;
    for (int32_t mi = 0; mi < mt; ++mi) {
      int32_t rows = m_i(p, mi);
      for (int32_t ni = 0; ni < nt; ++ni) {
        int32_t cols = n_i(p, ni);
        const int32_t *tile =
            (const int32_t *)(src + kc * plane + row_base + ni * c_tile_stride(p, mi));
        int32_t *dst = c + ((int64_t)kc * p->m + mi * p->tm) * p->n + ni * p->tn;
        for (int32_t r = 0; r < rows; ++r)
          memcpy(dst + (int64_t)r * p->n, tile + r * cols, (size_t)cols * 4);
      }
      row_base += nt * c_tile_stride(p, mi);
    }
  }
}

/* --- simulator backend (port of src/isa/sim.py) ------------------------- */
static uint8_t *sim_host(uint32_t phys, uint32_t bytes) {
  for (int32_t i = 0; i < rt.nregions; ++i) {
    npu_region *r = &rt.regions[i];
    if (phys >= r->phys && phys - r->phys + bytes <= r->size)
      return r->host + (phys - r->phys);
  }
  return 0;
}

static int32_t sim_run(const uint64_t *words, int32_t length, uint32_t data_base) {
  uint32_t m = 0, n = 0, k = 0, a = 0, b = 0, c = 0, ai = 0, bi = 0, ci = 0;
  for (int32_t pc = 0; pc < length; ++pc) {
    uint64_t w = words[pc];
    uint32_t op = (uint32_t)(w >> 56), lo = (uint32_t)w, hi = (uint32_t)(w >> 32);
    uint32_t count = 1;
    if (op == 0x15) { /* REPEAT */
      count = lo ? lo : 1;
      if (++pc >= length) return 0x15;
      w = words[pc];
      op = (uint32_t)(w >> 56); lo = (uint32_t)w; hi = (uint32_t)(w >> 32);
    }
    for (uint32_t rep = 0; rep < count; ++rep) {
      switch (op) {
      case 0x00: case 0x02: break;
      case 0x01: return 0;
      case 0x10: m = lo >> 24; n = (lo >> 16) & 0xff; k = lo & 0xffff; break;
      case 0x11: a = lo; break;
      case 0x12: b = lo; break;
      case 0x13: c = lo; break;
      case 0x14: ai = (lo & 0xffff) * 8; bi = (lo >> 16) * 8; ci = (hi & 0xffff) * 8; break;
      case 0x20: {
        if (!m || !n || !k || m > 16 || n > 16 || k > 256) return 0x11;
        if ((a | b | c) & 7) return 0x12;
        const int8_t *A = (const int8_t *)sim_host(data_base + a, m * k);
        const int8_t *B = (const int8_t *)sim_host(data_base + b, k * n);
        int32_t *C = (int32_t *)sim_host(data_base + c, 4 * m * n);
        if (!A || !B || !C) return 0x13;
        for (uint32_t i = 0; i < m; ++i)
          for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = 0;
            for (uint32_t q = 0; q < k; ++q)
              acc += (int32_t)A[i * k + q] * (int32_t)B[q * n + j];
            C[i * n + j] = acc;
          }
        rt.jobs += 1;
        rt.macs += (int64_t)m * n * k;
        if (lo & 1) a += ai;
        if (lo & 2) b += bi;
        if (lo & 4) c += ci;
        break;
      }
      default: return 0x10;
      }
    }
  }
  return 0x15; /* ran past the end */
}

/* --- PYNQ backend ------------------------------------------------------- */
static int32_t pynq_run(const npu_program *p) {
  volatile uint32_t *r = rt.mmio;
  barrier();
  r[REG_PROG_ADDR / 4] = rt.words_phys + 8u * (uint32_t)p->word_offset;
  r[REG_PROG_LEN / 4] = (uint32_t)p->length;
  r[REG_DATA_BASE / 4] = rt.io_phys;
  barrier();
  r[REG_ISA_CONTROL / 4] = 1;
  uint32_t status;
  do {
    status = r[REG_ISA_STATUS / 4];
    rt.polls += 1;
  } while (status & 0x9); /* RUNNING or BUSY */
  barrier();
  rt.cycles += r[REG_ISA_CYCLES / 4];
  rt.jobs += r[REG_ISA_JOBS / 4];
  rt.macs += (int64_t)p->m * p->n * p->k;
  if ((status & 0x4) || !(status & 0x2))
    return (int32_t)(r[REG_ISA_ERROR / 4] ? r[REG_ISA_ERROR / 4] : 0xff);
  return 0;
}

/* --- entry points ------------------------------------------------------- */
int32_t npu_rt_init(int32_t backend, volatile uint32_t *mmio, uint8_t *io,
                    uint32_t io_phys, uint32_t io_size, const uint64_t *words,
                    uint32_t words_phys, uint8_t *stage,
                    const npu_program *programs, int32_t count) {
  memset(&rt, 0, sizeof(rt));
  rt.backend = backend;
  rt.mmio = mmio;
  rt.io = io;
  rt.io_phys = io_phys;
  rt.io_size = io_size;
  rt.words = words;
  rt.words_phys = words_phys;
  rt.stage = stage;
  rt.programs = programs;
  rt.count = count;
  rt.error_program = -1;
  rt.regions[rt.nregions++] = (npu_region){io_phys, io, io_size};
  return 0;
}

/* Simulator only: a further physical range the programs may address (the
 * weight arena or the stage window). */
int32_t npu_rt_add_region(uint32_t phys, uint8_t *host, uint32_t size) {
  if (rt.nregions >= 4) return -1;
  rt.regions[rt.nregions++] = (npu_region){phys, host, size};
  return 0;
}

npu_rt_state *npu_rt_stats(void) { return &rt; }

void _mlir_ciface_npu_rt_gemm(int32_t id, memref_i8 *a, memref_i32 *c) {
  if (rt.error || id < 0 || id >= rt.count) {
    if (!rt.error) { rt.error = 0xfe; rt.error_program = id; }
    return;
  }
  const npu_program *p = &rt.programs[id];
  if (p->stage_src)
    memcpy(rt.stage, p->stage_src, (size_t)p->b_bytes);
  pack_a(p, a->aligned + a->offset, rt.io);
  int32_t status = rt.backend == NPU_RT_PYNQ
                       ? pynq_run(p)
                       : sim_run(rt.words + p->word_offset, p->length, rt.io_phys);
  rt.calls += 1;
  if (status) {
    rt.error = status;
    rt.error_program = id;
    return;
  }
  unpack_c(p, rt.io + p->c_base, c->aligned + c->offset);
}

/* --- MLIR runtime support ----------------------------------------------- */
/* memref.copy between non-contiguous memrefs lowers to a call of this
 * function (normally provided by MLIR's c_runner_utils). Descriptors use the
 * index width of the target, which is intptr_t on both of ours. */
typedef struct {
  intptr_t rank;
  void *descriptor;
} unranked_memref;

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
  /* Copy innermost runs with memcpy when they are contiguous on both sides. */
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
