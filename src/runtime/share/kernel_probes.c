/* __prof_enter / __prof_exit: what a model.so exported with --profile-kernels
 * calls around every CPU kernel (src/export/backend/cpu/profiling/kernel_probes.cpp).
 *
 *   void __prof_enter(int32_t id);
 *   void __prof_exit(int32_t id);
 *
 * Loaded with RTLD_GLOBAL before the model, as dispatch.c is. Off until
 * kp_enable(1): disabled, each probe is one load and one branch. Enabled, it
 * reads CLOCK_MONOTONIC (the clock time.perf_counter_ns and libnpu_rt use, so
 * all share one timeline) and adds the time to the kernel's counters; with an
 * event capacity it also keeps (id, begin, end) events until full, counting
 * those that do not fit.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEPTH 16

typedef struct {
  int64_t calls, total, min, max;
} stats;

static stats *table;
static int32_t kernels;
static int enabled;
static int depth;
static int32_t open_id[DEPTH];
static int64_t open_ns[DEPTH];
static int64_t *events;   /* 3 per event: id, begin_ns, end_ns */
static int64_t capacity, recorded, dropped;

static int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000 + ts.tv_nsec;
}

/* Clear every counter for `count` kernels and keep up to `event_capacity`
 * events. 0 on success. Leaves profiling off. */
int kp_reset(int32_t count, int64_t event_capacity) {
  enabled = 0;
  free(table);
  free(events);
  table = calloc(count > 0 ? (size_t)count : 1, sizeof(stats));
  events = event_capacity > 0 ? malloc((size_t)event_capacity * 3 * sizeof(int64_t)) : NULL;
  kernels = table ? count : 0;
  capacity = events ? event_capacity : 0;
  recorded = dropped = 0;
  depth = 0;
  return table && (event_capacity <= 0 || events) ? 0 : -1;
}

void kp_enable(int on) {
  depth = 0;
  enabled = on && table;
}

/* calls, total_ns, min_ns, max_ns of kernel `id`. 0 on success. */
int kp_stats(int32_t id, int64_t out[4]) {
  if (id < 0 || id >= kernels)
    return -1;
  out[0] = table[id].calls;
  out[1] = table[id].total;
  out[2] = table[id].min;
  out[3] = table[id].max;
  return 0;
}

/* Copies up to `max` events (3 values each) into out; returns how many were
 * recorded. *lost gets the number that did not fit. */
int64_t kp_events(int64_t *out, int64_t max, int64_t *lost) {
  int64_t n = recorded < max ? recorded : max;
  if (out && n > 0)
    memcpy(out, events, (size_t)n * 3 * sizeof(int64_t));
  if (lost)
    *lost = dropped;
  return recorded;
}

void __prof_enter(int32_t id) {
  if (!enabled)
    return;
  if (depth < DEPTH) {
    open_id[depth] = id;
    open_ns[depth] = now_ns();
  }
  depth += 1;
}

void __prof_exit(int32_t id) {
  if (!enabled || depth == 0)
    return;
  depth -= 1;
  if (depth >= DEPTH || open_id[depth] != id || id < 0 || id >= kernels)
    return;
  int64_t end = now_ns(), ns = end - open_ns[depth];
  stats *s = &table[id];
  s->min = s->calls ? (ns < s->min ? ns : s->min) : ns;
  s->max = ns > s->max ? ns : s->max;
  s->calls += 1;
  s->total += ns;
  if (recorded < capacity) {
    events[recorded * 3] = id;
    events[recorded * 3 + 1] = open_ns[depth];
    events[recorded * 3 + 2] = end;
    recorded += 1;
  } else if (capacity) {
    dropped += 1;
  }
}
