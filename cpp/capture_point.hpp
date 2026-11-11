// capture_point.hpp: header-only C++17 port of breakpoint/humanoid.py (biped push recovery).
//
// Linear inverted pendulum + capture-point stepping, with ankle-torque, friction, swing-time,
// leg-reach, lateral-reach and latency limits; latency is compensated by a Smith predictor.
// Allocation-free inside the control step: the latency history and the commanded-CoP history are
// fixed ring buffers sized at construction, so `step()` is safe to call from a real-time loop. The
// batch `recover()` mirrors the Python reference exactly and is checked against it by tests/test_cpp.py.
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
    // Latency in ticks, rounding halves up like humanoid.py; a negative latency is treated as none.
    lat_ = static_cast<std::size_t>(std::max(0.0, std::floor(c.latency_s / kDt + 0.5)));
    ring_.assign(lat_ + 1, Vec2{});
    // Smith predictor state (see humanoid.py): commanded-CoP ring and the running sum of its terms.
    hx_ = std::min(r.foot_half_length, cop_reach_);
    e_dt_ = std::exp(w_ * kDt);
    e_n_ = std::pow(e_dt_, static_cast<double>(lat_));
    const double shw = r.stance_half_width + r.foot_half_width;
    const Vec2 p0{std::clamp(pre_push_.x, -hx_, hx_), std::clamp(pre_push_.y, -shw, shw)};
    cop_ring_.assign(lat_, p0);
    cop_sum_ = p0 * (e_n_ - 1.0);
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
    const Vec2 est = lat_ ? seen * e_n_ - cop_sum_ : seen;  // best estimate of the capture point now

    const double hw = dbl_ ? r_.stance_half_width + r_.foot_half_width : r_.foot_half_width;
    const double hx = hx_, fw = r_.foot_half_width;
    const Vec2 lo{stance_.x - hx, stance_.y - hw}, hi{stance_.x + hx, stance_.y + hw};
    Vec2 b0 = lo, b1 = hi;
    if (planned_ && dbl_) {  // first swing: only the other foot can hold the CoP
      b0 = {swing_stance_.x - hx, swing_stance_.y - fw};
      b1 = {swing_stance_.x + hx, swing_stance_.y + fw};
    }
    Vec2 p{std::clamp(est.x, b0.x, b1.x), std::clamp(est.y, b0.y, b1.y)};
    // friction cone: pull the CoP toward the CoM until |x - p| <= mu z; slip only if no support point can
    const double r = c_.friction * z_;
    const Vec2 q{std::clamp(x_.x, b0.x, b1.x), std::clamp(x_.y, b0.y, b1.y)};
    if ((x_ - q).norm() > r) return finish(false, "slipped");
    if ((x_ - p).norm() > r) {
      const Vec2 d = p - q, f = q - x_;
      const double a = d.x * d.x + d.y * d.y, b = 2 * (f.x * d.x + f.y * d.y), cc = f.x * f.x + f.y * f.y - r * r;
      const double sr = std::clamp((-b + std::sqrt(std::max(b * b - 4 * a * cc, 0.0))) / (2 * a), 0.0, 1.0);
      p = q + d * sr;
    }
    if (lat_) {
      const std::size_t j = tick_ % lat_;
      cop_sum_ = cop_sum_ * e_dt_ + p * (e_dt_ - 1.0) - cop_ring_[j] * (e_n_ * (e_dt_ - 1.0));
      cop_ring_[j] = p;
    }
    ++tick_;
    const bool inside = est.x >= lo.x - 1e-9 && est.x <= hi.x + 1e-9 && est.y >= lo.y - 1e-9 && est.y <= hi.y + 1e-9;
    if (!planned_ && !inside) {
      if (steps_ >= r_.max_steps) return finish(false, "needed more than max steps");
      Vec2 q = p;  // CoP during the coming swing
      if (dbl_) {
        const double side = est.y - stance_.y > 1e-9 ? 1.0 : est.y - stance_.y < -1e-9 ? -1.0 : 1.0;
        swing_stance_ = {stance_.x, stance_.y - side * r_.stance_half_width};
        q = {std::clamp(est.x, swing_stance_.x - hx, swing_stance_.x + hx), std::clamp(est.y, swing_stance_.y - fw, swing_stance_.y + fw)};
      }
      const double e = std::exp(w_ * swing_);
      Vec2 d = q + (est - q) * e - stance_;
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
    const double rw = dbl_ ? r_.stance_half_width + r_.foot_half_width : r_.foot_half_width;
    if (!planned_ && v_.norm() < 0.02 && std::abs(nxi.x - stance_.x) <= hx_ && std::abs(nxi.y - stance_.y) <= rw && t_ > 0.3)
      return finish(true, "came to rest");
    if ((nxi - stance_).norm() > r_.max_step * 3) return finish(false, "capture point ran away");
  }

  // Run to completion (the batch reference used by the tests).
  Result run(double horizon_s = 5.0) {
    while (!done_ && t_ < horizon_s) step();
    if (!done_) {
      const double hw = dbl_ ? r_.stance_half_width + r_.foot_half_width : r_.foot_half_width;
      const Vec2 xi = capture_point();
      const bool ok = std::abs(xi.x - stance_.x) <= hx_ && std::abs(xi.y - stance_.y) <= hw;
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
  double hx_ = 0, e_dt_ = 1, e_n_ = 1;
  Vec2 x_, v_, stance_, target_, pre_push_, swing_stance_, cop_sum_;
  bool dbl_ = true, planned_ = false, done_ = false;
  int steps_ = 0;
  std::size_t lat_ = 0, head_ = 0, count_ = 0, tick_ = 0;
  std::vector<Vec2> ring_;      // measured capture points; sized once in the constructor, never grows in step()
  std::vector<Vec2> cop_ring_;  // commanded CoP over the last `lat_` ticks; same rule
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
