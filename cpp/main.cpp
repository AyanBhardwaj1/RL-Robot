// breakpoint_cp: command-line front end and real-time loop benchmark for capture_point.hpp.
//
//   breakpoint_cp recover <impulse> <direction_deg> [latency_s] [payload_kg] [friction]
//   breakpoint_cp envelope [latency_s]
//   breakpoint_cp bench [seconds]        # 500 Hz fixed-rate loop: step cost and wake-up jitter
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>

#include "capture_point.hpp"

using namespace breakpoint;

static Conditions cond_from(int argc, char** argv, int at) {
  Conditions c;
  if (argc > at) c.latency_s = std::atof(argv[at]);
  if (argc > at + 1) c.payload_kg = std::atof(argv[at + 1]);
  if (argc > at + 2) c.friction = std::atof(argv[at + 2]);
  return c;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s recover|envelope|bench ...\n", argv[0]);
    return 2;
  }
  if (!std::strcmp(argv[1], "recover") && argc >= 4) {
    const Result r = recover(std::atof(argv[2]), std::atof(argv[3]), Humanoid{}, cond_from(argc, argv, 4));
    std::printf("{\"recovered\": %s, \"steps\": %d, \"reason\": \"%s\", \"time_s\": %.3f}\n",
                r.recovered ? "true" : "false", r.steps, r.reason.c_str(), r.time_s);
    return 0;
  }
  if (!std::strcmp(argv[1], "envelope")) {
    const Conditions c = cond_from(argc, argv, 2);
    const char* names[] = {"front", "front-left", "left", "rear-left", "rear", "rear-right", "right", "front-right"};
    const double dirs[] = {180, -135, -90, -45, 0, 45, 90, 135};
    std::printf("{");
    for (int i = 0; i < 8; ++i) std::printf("%s\"%s\": %.2f", i ? ", " : "", names[i], breaking_point(dirs[i], Humanoid{}, c));
    std::printf("}\n");
    return 0;
  }
  if (!std::strcmp(argv[1], "bench")) {
    // A 500 Hz loop like a real controller: each tick runs one controller step, then sleeps to the next deadline.
    const double seconds = argc > 2 ? std::atof(argv[2]) : 2.0;
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::microseconds(2000);
    std::vector<double> step_us, jitter_us;
    CapturePointController ctl(60.0, -90.0);
    auto next = clock::now() + period;
    const auto end = clock::now() + std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(seconds));
    while (clock::now() < end) {
      if (ctl.done()) ctl = CapturePointController(60.0, -90.0);  // restart the push scenario (allocates; not timed)
      const auto t0 = clock::now();
      ctl.step();
      step_us.push_back(std::chrono::duration<double, std::micro>(clock::now() - t0).count());
      std::this_thread::sleep_until(next);
      jitter_us.push_back(std::chrono::duration<double, std::micro>(clock::now() - next).count());
      next += period;
    }
    auto pct = [](std::vector<double> v, double q) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(q * v.size()))]; };
    std::printf("{\"ticks\": %zu, \"step_us_p50\": %.2f, \"step_us_p99\": %.2f, \"step_us_max\": %.2f, "
                "\"wake_jitter_us_p50\": %.1f, \"wake_jitter_us_p99\": %.1f, \"wake_jitter_us_max\": %.1f}\n",
                step_us.size(), pct(step_us, .5), pct(step_us, .99), *std::max_element(step_us.begin(), step_us.end()),
                pct(jitter_us, .5), pct(jitter_us, .99), *std::max_element(jitter_us.begin(), jitter_us.end()));
    return 0;
  }
  std::fprintf(stderr, "unknown command\n");
  return 2;
}
