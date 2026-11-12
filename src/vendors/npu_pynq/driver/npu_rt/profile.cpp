/* Phase profiling: where one delegate call's time goes.
 *
 * npu_rt_execute times packA, each segment's weight copy into the stage
 * buffer, each segment's run (start to done on the board, the ISA simulator on
 * the host) and unpackC, and adds them to the program's PhaseStats. Off by
 * default; disabled, execute does one branch per call and nothing else.
 */
#include "npu_rt.h"

#include <algorithm>
#include <ctime>

using namespace npu_rt;

int64_t npu_rt::nowNs() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000 + ts.tv_nsec;
}

void npu_rt::recordEvent(int64_t index, int64_t phase, int64_t begin, int64_t end) {
  if (rt.events.size() < rt.eventCapacity)
    rt.events.push_back({index, phase, begin, end});
  else if (rt.eventCapacity)
    rt.eventsDropped += 1;
}

extern "C" void npu_rt_profile(int enable, int64_t event_capacity) {
  rt.profile = enable != 0;
  rt.phases.assign(enable ? rt.programs.size() : 0, PhaseStats{});
  rt.events.clear();
  rt.eventsDropped = 0;
  rt.eventCapacity = enable && event_capacity > 0 ? static_cast<size_t>(event_capacity) : 0;
  if (rt.eventCapacity)
    rt.events.reserve(rt.eventCapacity);
  else
    rt.events.shrink_to_fit();
}

extern "C" int64_t npu_rt_program_count(void) { return static_cast<int64_t>(rt.programs.size()); }

extern "C" int npu_rt_phase_stats(int64_t index, int64_t out[NPU_RT_PHASE_FIELDS]) {
  if (index < 0 || index >= static_cast<int64_t>(rt.programs.size()))
    return -1;
  const Program &p = rt.programs[index];
  PhaseStats s = index < static_cast<int64_t>(rt.phases.size()) ? rt.phases[index] : PhaseStats{};
  const int64_t values[NPU_RT_PHASE_FIELDS] = {
      p.m, p.n, p.k, static_cast<int64_t>(p.segments.size()), s.calls, s.callMin, s.callMax,
      s.packA, s.stageB, s.pl, s.unpackC, s.plCycles, s.polls, s.jobs,
      s.packABytes, s.stageBBytes, s.unpackCBytes, s.runs, s.instructions};
  std::copy(values, values + NPU_RT_PHASE_FIELDS, out);
  return 0;
}

extern "C" int64_t npu_rt_events(int64_t *out, int64_t max, int64_t *dropped) {
  int64_t n = std::min<int64_t>(max, static_cast<int64_t>(rt.events.size()));
  for (int64_t i = 0; i < n; ++i) {
    const Event &e = rt.events[i];
    int64_t *o = out + i * NPU_RT_EVENT_FIELDS;
    o[0] = e.index;
    o[1] = e.phase;
    o[2] = e.begin;
    o[3] = e.end;
  }
  if (dropped)
    *dropped = rt.eventsDropped;
  return static_cast<int64_t>(rt.events.size());
}
