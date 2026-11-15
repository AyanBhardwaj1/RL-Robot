# C++ capture-point controller

`capture_point.hpp` is a header-only C++17 port of `breakpoint/humanoid.py`'s push recovery for all three
bodies. It covers:

- linear inverted pendulum dynamics,
- capture-point stepping for the biped, with ankle-torque, friction cone, swing-time, stance-leg,
  leg-reach and lateral-reach limits,
- control latency, compensated by a Smith predictor,
- the wheeled base and the wheel-legged hybrid (`WheeledController`, `wheel_legged()`): the base drives
  under the capture point with bounded acceleration and speed, a reaction delay, a tip-over limit and a
  step-up height.

`CapturePointController::step()` and `WheeledController::step()` each advance one 2 ms tick. They allocate
nothing and throw nothing: the latency histories are ring buffers sized once in the constructor. `tests/test_cpp.py` compiles the code with
`-Wall -Wextra -Wpedantic -Werror` and checks every case against the Python reference. The 16
breaking-point values match to 0.01 N·s.

```bash
cmake -S cpp -B build-cpp && cmake --build build-cpp
./build-cpp/breakpoint_cp recover 60 -90        # 60 N·s shove from the left
./build-cpp/breakpoint_cp recover-wheeled 90 45 0 0 0.8 wheel-legged   # the hybrid, shoved from the rear-right
./build-cpp/breakpoint_cp envelope 0.2          # breaking points with 200 ms of latency
./build-cpp/breakpoint_cp bench 2               # 500 Hz loop: step cost and wake-up jitter
```

## What the benchmark shows

Five runs on a stock (non-real-time) Linux container, 500 Hz, 3 s each; median across runs, with the
range in brackets. The step includes the Smith predictor and the friction-cone CoP.

| Metric | p50 | p99 | max |
|---|---|---|---|
| Controller step | 0.7 µs (0.66–0.73) | 2.2 µs (1.6–3.4) | 21 µs (20–96) |
| Wake-up jitter after `sleep_until` | 124 µs (112–149) | 0.87 ms (0.37–2.1) | 6.6 ms (1.9–15.4) |

The control math costs well under 0.1% of the 2 ms budget. The risk is the scheduler: the worst late
wake-up in a 3 s run ranged from 1.9 ms to 15 ms, i.e. up to seven whole control periods missed. The
spread between runs is itself the finding: on a stock kernel the worst case is not even repeatable. That is why real-time controllers run on a PREEMPT_RT kernel
with an isolated core and a real-time scheduling class, and why the humanoid lab treats latency as a
stress axis.
