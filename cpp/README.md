# C++ capture-point controller

`capture_point.hpp` is a header-only C++17 port of `breakpoint/humanoid.py`'s biped push recovery. It covers:

- linear inverted pendulum dynamics,
- capture-point stepping,
- ankle-torque, friction, swing-time, leg-reach and lateral-reach limits,
- control latency.

`CapturePointController::step()` advances one 2 ms tick. It allocates nothing and throws nothing: the
latency history is a ring buffer sized once in the constructor. `tests/test_cpp.py` compiles the code with
`-Wall -Wextra -Wpedantic -Werror` and checks every case against the Python reference. The 16
breaking-point values match to 0.01 N·s.

```bash
cmake -S cpp -B build-cpp && cmake --build build-cpp
./build-cpp/breakpoint_cp recover 60 -90        # 60 N·s shove from the left
./build-cpp/breakpoint_cp envelope 0.2          # breaking points with 200 ms of latency
./build-cpp/breakpoint_cp bench 2               # 500 Hz loop: step cost and wake-up jitter
```

## What the benchmark shows

Run on a stock (non-real-time) Linux container, 500 Hz, 2 s:

| Metric | p50 | p99 | max |
|---|---|---|---|
| Controller step | 0.8 µs | 1.7 µs | 8.6 µs |
| Wake-up jitter after `sleep_until` | 134 µs | 423 µs | 4.9 ms |

The control math costs about 0.1% of the 2 ms budget. The risk is the scheduler: a 4.9 ms late wake-up
misses more than two whole control periods. That is why real-time controllers run on a PREEMPT_RT kernel
with an isolated core and a real-time scheduling class, and why the humanoid lab treats latency as a
stress axis.
