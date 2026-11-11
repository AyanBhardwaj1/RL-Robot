// capture_point.hpp: header-only C++17 port of breakpoint/humanoid.py (biped push recovery).
//
// Linear inverted pendulum + capture-point stepping, with ankle-torque, friction, swing-time,
// leg-reach, lateral-reach and latency limits. Allocation-free inside the control step: the
// latency history is a fixed ring buffer sized at construction, so `step()` is safe to call
// from a real-time loop. The batch `recover()` mirrors the Python reference exactly and is
// checked against it by tests/test_cpp.py.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace breakpoint {

constexpr double kG = 9.81;
constexpr double kDt = 0.002;
constexpr double kPi = 3.14159265358979323846;

struct Vec2 {
  double x = 0, y = 0;
  Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
  Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
  Vec2 operator*(double s) const { return {x * s, y * s}; }
  double norm() const { return std::hypot(x, y); }
};

struct Humanoid {
  double mass_kg = 72.0, com_height = 0.95;
  double foot_half_length = 0.11, foot_half_width = 0.05, stance_half_width = 0.11;
  double max_step = 0.55, max_lateral_step = 0.40, min_swing = 0.32;
  int max_steps = 3;
  double max_step_up = 0.18, ankle_torque = 150.0, swing_load_factor = 0.6;
};

struct Conditions {
  double payload_kg = 0, payload_reach = 0.35, payload_height = 1.05, friction = 0.8;
  double latency_s = 0, actuator_strength = 1.0, floor_slope_deg = 0, floor_step_m = 0;
};

struct Result {
  bool recovered = false;
  int steps = 0;
  std::string reason;
  double time_s = 0;
};

// One controller instance per push. step() advances 2 ms; done() reports the outcome.
class CapturePointController {
 public:
  CapturePointController(double impulse, double direction_deg, const Humanoid& r = {}, const Conditions& c = {})
      : r_(r), c_(c) {
    if (c.floor_step_m > r.max_step_up) {
      result_ = {false, 0, "blocked", 0};
      done_ = true;
      return;
    }
    const double m = r.mass_kg + c.payload_kg;
    z_ = (r.mass_kg * r.com_height + c.payload_kg * c.payload_height) / m;
    w_ = std::sqrt(kG / z_);
    const double strength = std::max(c.actuator_strength, 0.05);
    swing_ = r.min_swing * (1 + r.swing_load_factor * c.payload_kg / r.mass_kg) / strength;
    slope_acc_ = kG * std::sin(c.floor_slope_deg * kPi / 180.0);
    bias_x_ = slope_acc_ / (w_ * w_);
    cop_reach_ = r.ankle_torque * strength / (m * kG);
    x_ = {c.payload_kg * c.payload_reach / m, 0.0};
    pre_push_ = x_ + Vec2{bias_x_, 0};
    const double a = direction_deg * kPi / 180.0;
    v_ = Vec2{std::cos(a), std::sin(a)} * (impulse / m);
    lat_ = static_cast<std::size_t>(std::lround(c.latency_s / kDt));
    ring_.assign(lat_ + 1, Vec2{});
  }

  bool done() const { return done_; }
  const Result& result() const { return result_; }
  Vec2 com() const { return x_; }
  Vec2 capture_point() const { return x_ + v_ * (1.0 / w_) + Vec2{bias_x_, 0}; }

  // Advance one 2 ms tick. No allocation, no exceptions.
  void step() {
    if (done_) return;
    const Vec2 xi = capture_point();
    // latency: the controller acts on the capture point from `lat_` ticks ago (the first sample until then)
    ring_[head_] = xi;
    ++count_;
    const Vec2 seen = count_ > lat_ ? ring_[(head_ + 1) % ring_.size()] : pre_push_;  // stale view = standing still
    head_ = (head_ + 1) % ring_.size();

    const double hw = dbl_ ? r_.stance_half_width + r_.foot_half_width : r_.foot_half_width;
    const double hx = std::min(r_.foot_half_length, cop_reach_);
    const Vec2 lo{stance_.x - hx, stance_.y - hw}, hi{stance_.x + hx, stance_.y + hw};
    const Vec2 p{std::clamp(xi.x, lo.x, hi.x), std::clamp(xi.y, lo.y, hi.y)};
    const double dn = (x_ - p).norm();
    if (dn > c_.friction * z_) return finish(false, "slipped");
    const bool inside = seen.x >= lo.x - 1e-9 && seen.x <= hi.x + 1e-9 && seen.y >= lo.y - 1e-9 && seen.y <= hi.y + 1e-9;
    if (!planned_ && !inside) {
      if (steps_ >= r_.max_steps) return finish(false, "needed more than max steps");
      const double e = std::exp(w_ * swing_);
      Vec2 d = p + (seen - p) * e - stance_;
      const double reach = d.norm();
      if (reach > r_.max_step) d = d * (r_.max_step / reach);
      d.y = std::clamp(d.y, -r_.max_lateral_step, r_.max_lateral_step);
      target_ = stance_ + d;
      planned_ = true;
      swing_until_ = t_ + swing_;
    }
    if (planned_ && t_ >= swing_until_) {
      stance_ = target_;
      planned_ = false;
      dbl_ = false;
      ++steps_;
    }
    const Vec2 acc = (x_ - p) * (w_ * w_) + Vec2{slope_acc_, 0};
    v_ = v_ + acc * kDt;
    x_ = x_ + v_ * kDt;
    t_ += kDt;
    const Vec2 nxi = capture_point();
    if (!planned_ && v_.norm() < 0.02 && std::abs(nxi.x - stance_.x) <= r_.foot_half_length + 1e-9 &&
        std::abs(nxi.y - stance_.y) <= r_.foot_half_width + 1e-9 && t_ > 0.3)
      return finish(true, "came to rest");
    if ((nxi - stance_).norm() > r_.max_step * 3) return finish(false, "capture point ran away");
  }

  // Run to completion (the batch reference used by the tests).
  Result run(double horizon_s = 5.0) {
    while (!done_ && t_ < horizon_s) step();
    if (!done_) {
      const double hw = dbl_ ? r_.stance_half_width + r_.foot_half_width : r_.foot_half_width;
      const double hx = std::min(r_.foot_half_length, cop_reach_);
      const Vec2 xi = capture_point();
      const bool ok = std::abs(xi.x - stance_.x) <= hx && std::abs(xi.y - stance_.y) <= hw;
      finish(ok, ok ? "came to rest" : "still moving");
    }
    return result_;
  }

 private:
  void finish(bool ok, const char* why) {
    result_ = {ok, steps_, why, t_};
    done_ = true;
  }

  Humanoid r_;
  Conditions c_;
  double z_ = 0, w_ = 1, swing_ = 0, slope_acc_ = 0, bias_x_ = 0, cop_reach_ = 0, t_ = 0, swing_until_ = -1;
  Vec2 x_, v_, stance_, target_, pre_push_;
  bool dbl_ = true, planned_ = false, done_ = false;
  int steps_ = 0;
  std::size_t lat_ = 0, head_ = 0, count_ = 0;
  std::vector<Vec2> ring_;  // sized once in the constructor; never grows in step()
  Result result_;
};

inline Result recover(double impulse, double direction_deg, const Humanoid& r = {}, const Conditions& c = {}) {
  return CapturePointController(impulse, direction_deg, r, c).run();
}

inline double breaking_point(double direction_deg, const Humanoid& r = {}, const Conditions& c = {},
                             double max_impulse = 200.0, int iters = 12) {
  auto ok = [&](double j) { return recover(j, direction_deg, r, c).recovered; };
  if (!ok(0)) return 0;
  if (ok(max_impulse)) return max_impulse;
  double lo = 0, hi = max_impulse;
  for (int k = 0; k < iters; ++k) {
    const double mid = 0.5 * (lo + hi);
    (ok(mid) ? lo : hi) = mid;
  }
  return std::round(lo * 100) / 100;
}

}  // namespace breakpoint
