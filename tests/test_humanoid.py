import math

import pytest

from breakpoint.humanoid import (
    DIRECTIONS,
    JOBS,
    Conditions,
    Humanoid,
    WheeledBase,
    breaking_point,
    effective,
    job_risk,
    recover,
    recover_wheeled,
    zero_step_limit,
)


def test_standing_still_is_stable_for_both_bodies():
    assert recover(0, 0).recovered and recover(0, 0).steps == 0
    assert recover_wheeled(0, 0).recovered


def test_small_push_uses_ankles_only_and_matches_capture_point_theory():
    lim = zero_step_limit(0.0)  # forward push: capture point reaches the toe
    assert recover(0.8 * lim, 0.0).steps == 0
    r = recover(1.3 * lim, 0.0)
    assert r.recovered and r.steps >= 1
    # analytic: impulse = m * w * distance_to_edge
    robot = Humanoid()
    assert lim == pytest.approx(robot.mass_kg * robot.omega * robot.foot_half_length, rel=1e-6)


def test_bigger_pushes_need_more_steps_then_fail():
    steps = [recover(j, 0.0).steps for j in (20, 50, 85)]
    assert steps == sorted(steps)
    assert not recover(160, 0.0).recovered


def test_latency_shrinks_the_envelope():
    fast = breaking_point(-90.0, cond=Conditions(latency_s=0.0), iters=8)
    slow = breaking_point(-90.0, cond=Conditions(latency_s=0.2), iters=8)
    assert slow < 0.8 * fast


def test_tote_in_front_hurts_forward_pushes_more_than_backward():
    cond = Conditions(payload_kg=20)
    forward = breaking_point(DIRECTIONS["rear"], cond=cond, iters=8)   # shove from the rear -> falls forward
    backward = breaking_point(DIRECTIONS["front"], cond=cond, iters=8)
    assert forward < backward
    z, offset, _ = effective(Humanoid(), cond)
    assert offset == pytest.approx(20 * 0.35 / 92) and z > Humanoid().com_height


def test_low_friction_makes_it_slip():
    r = recover(70, 0.0, cond=Conditions(friction=0.15))
    assert not r.recovered and r.reason.startswith("slipped")


def test_slope_is_compensated_by_leaning():
    assert recover(0, 0, cond=Conditions(floor_slope_deg=4)).recovered


def test_wheeled_base_resists_shoves_but_cannot_climb():
    assert breaking_point(-90.0, WheeledBase(), iters=8) > breaking_point(-90.0, Humanoid(), iters=8)
    blocked = recover_wheeled(0, 0, cond=Conditions(floor_step_m=0.10))
    assert not blocked.recovered and blocked.reason.startswith("blocked")
    assert recover(0, 0, cond=Conditions(floor_step_m=0.10)).recovered


def test_job_risk_reports_rate_bound_and_causes():
    r = job_risk("home care", WheeledBase(), trials=40, seed=1)
    assert r["trials"] == 40 and r["failures"] > 0 and "blocked" in r["causes"]
    assert r["failure_rate_upper_95"] >= r["failure_rate"]
    with pytest.raises(ValueError):
        job_risk("moonwalk")
    assert set(JOBS) >= {"line-side kitting", "tote handling", "home care"}


def test_recording_produces_footsteps_and_paths():
    r = recover(60, -90.0, record=True)
    assert r.footsteps and r.com and r.capture
    assert all(math.isfinite(v) for row in r.com for v in row)
