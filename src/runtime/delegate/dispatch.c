/* __delegate__: what model.so calls for every region a delegate runs.
 *
 * The back end replaced each claimed region with
 *
 *   void _mlir_ciface___delegate__(int64_t id, memref_1d_i8 *in, memref_1d_i8 *out)
 *
 * This library defines it, loaded with RTLD_GLOBAL before the model, and
 * forwards the call to the backend registered for that id (for the NPU,
 * libnpu_rt's npu_rt_execute) with the device's own index for it.
 */
#include <stdint.h>
#include <stdlib.h>

typedef int (*delegate_fn)(int64_t index, void **buffers, int count);

typedef struct {
  int8_t *allocated, *aligned;
  intptr_t offset, size, stride;
} memref_i8;

typedef struct {
  delegate_fn fn;
  int64_t index;
} entry;

static entry *table;
static int64_t entries;
static int64_t calls;
static int status, failed_id = -1;

int delegate_reset(int64_t count) {
  free(table);
  table = calloc(count > 0 ? (size_t)count : 1, sizeof(entry));
  entries = table ? count : 0;
  calls = 0;
  status = 0;
  failed_id = -1;
  return table ? 0 : -1;
}

int delegate_register(int64_t id, delegate_fn fn, int64_t index) {
  if (id < 0 || id >= entries)
    return -1;
  table[id].fn = fn;
  table[id].index = index;
  return 0;
}

/* The first failure since the last delegate_clear(), and the id it came from. */
int delegate_status(int64_t *id) {
  if (id)
    *id = failed_id;
  return status;
}

void delegate_clear(void) {
  status = 0;
  failed_id = -1;
}

int64_t delegate_calls(void) { return calls; }

void _mlir_ciface___delegate__(int64_t id, memref_i8 *in, memref_i8 *out) {
  if (status)
    return;   /* the model keeps running; the caller checks delegate_status() */
  if (id < 0 || id >= entries || !table[id].fn) {
    status = -1;
    failed_id = id;
    return;
  }
  void *buffers[2] = {in->aligned + in->offset, out->aligned + out->offset};
  int rc = table[id].fn(table[id].index, buffers, 2);
  calls += 1;
  if (rc) {
    status = rc;
    failed_id = id;
  }
}
