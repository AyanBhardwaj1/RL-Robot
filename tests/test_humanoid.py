import itertools
import math

import numpy as np
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


def test_hazard_classification_and_risk_matrix():
    from breakpoint.hazards import HAZARDS, RISK, classify, likelihood_band, to_markdown

    assert classify("slipped: needed friction 0.4 > 0.3").hid == "H2"
    assert classify("blocked: a 10 cm step").hid == "H4"
    assert classify("needed more than 3 steps").hid == "H1"
    assert classify("came to rest") is None
    assert [likelihood_band(x) for x in (0.005, 0.03, 0.1, 0.5)] == [0, 1, 2, 3]
    assert RISK[4][3] == "high" and RISK[1][0] == "low"
    assert len({h.hid for h in HAZARDS}) == len(HAZARDS)
    md = to_markdown([], 10)
    assert "No failures recorded" in md


def test_hazard_log_reports_measured_rates():
    from breakpoint.hazards import hazard_log

    rows = hazard_log(trials=30, seed=2)
    assert rows, "the wheeled base should be blocked on some home-care routes"
    for r in rows:
        assert 0 <= r["rate"] <= r["rate_upper_95"] <= 1 and r["risk"] in ("low", "medium", "high")
    assert any(r["hazard"] == "H4" and r["body"] == "wheeled" for r in rows)


def test_fleet_fit_recovers_profile_and_ignores_glitches(tmp_path):
    import json as _json

    from breakpoint.fleet import fit_profile, read_log, site_odds, synthetic_log

    recs = synthetic_log("tote handling", shifts=400, seed=3)
    assert any(r["friction"] == 9.99 for r in recs)  # glitched readings are present
    prof = fit_profile(recs)
    true = JOBS["tote handling"]
    assert prof.friction[1] < 1.2  # quantiles ignore the 9.99 glitch
    assert abs(prof.payload[0] - true.payload[0]) < 2.5 and abs(prof.payload[1] - true.payload[1]) < 2.5
    assert abs(prof.push_prob - true.push_prob) < 0.08
    path = tmp_path / "log.jsonl"
    path.write_text("\n".join(_json.dumps(r) for r in recs[:50]) + "\n")
    assert len(read_log(path)) == 50
    res = site_odds(read_log(path), "t", trials=20)
    assert set(res["odds"]) == {"biped", "wheeled"} and "__fleet__t" not in JOBS
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n")
    with pytest.raises(ValueError):
        read_log(bad)
    with pytest.raises(ValueError):
        synthetic_log("moon")


def test_swing_foot_carries_no_weight_so_side_pushes_are_harder():
    """Regression: during the first swing the CoP must stay on the loaded foot (review finding, high)."""
    from breakpoint.humanoid import envelope

    env = envelope()
    assert env["left"] == env["right"] and env["front"] == env["rear"]
    assert env["left"] < 0.6 * env["front"]  # the stepping leg cannot cross the stance leg
    assert env["front-left"] < env["front"]


def test_latency_costs_ankle_and_stepping_smoothly():
    """Regression: the ankles also act on delayed information, and the cost falls smoothly with latency."""
    from breakpoint.humanoid import breaking_point

    def ankle_only(lat):
        lo, hi = 0.0, 60.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            r = recover(mid, 180.0, cond=Conditions(latency_s=lat))
            lo, hi = (mid, hi) if (r.recovered and r.steps == 0) else (lo, mid)
        return lo

    a0, a1, a2 = ankle_only(0.0), ankle_only(0.1), ankle_only(0.25)
    assert a0 > a1 > a2 > 0
    front = [breaking_point(180.0, cond=Conditions(latency_s=lat)) for lat in (0.0, 0.05, 0.1, 0.15, 0.25)]
    assert all(a > b for a, b in itertools.pairwise(front))
    w = math.sqrt(9.81 / 0.95)
    assert front[-1] / front[0] == pytest.approx(math.exp(-w * 0.25), abs=0.05)  # unseen for L seconds


def test_zero_step_limit_matches_simulation_with_torque_and_slope():
    """Regression: the analytic ankle-only limit uses the same torque-capped foot and slope lean as the sim."""
    from breakpoint.humanoid import zero_step_limit

    for cond in (Conditions(), Conditions(actuator_strength=0.5, payload_kg=20.0), Conditions(floor_slope_deg=3.0)):
        lo, hi = 0.0, 80.0
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            r = recover(mid, 180.0, cond=cond)
            lo, hi = (mid, hi) if (r.recovered and r.steps == 0) else (lo, mid)
        assert zero_step_limit(180.0, cond=cond) == pytest.approx(lo, rel=0.01), cond


def test_ankle_only_side_push_settles_quickly():
    """Regression: the rest check uses the double-support box, not one foot centred between the feet."""
    r = recover(30.0, -90.0)
    assert r.recovered and r.steps == 0 and r.time_s < 1.5


def test_latency_ticks_round_half_up_like_js_and_cpp():
    from breakpoint.humanoid import _ticks

    assert _ticks(0.005) == 3 and _ticks(0.025) == 13 and _ticks(0.0) == 0 and _ticks(-0.1) == 0


def test_slip_is_blamed_only_when_grip_would_have_saved_it():
    """A falling robot's foot always slips at the end; the cause must be the fall, not the floor."""
    ice = recover(60.0, 180.0, cond=Conditions(friction=0.1))
    assert not ice.recovered and ice.reason.startswith("slipped")
    fall = recover(61.0, 103.0, cond=Conditions(friction=0.5, floor_slope_deg=-2.7, payload_kg=15.0))
    assert not fall.recovered and not fall.reason.startswith("slipped")


def test_friction_cone_respected_by_the_controller():
    """The CoP is pulled toward the CoM rather than demanding more friction than the floor has."""
    from breakpoint.humanoid import _friction_limited

    box = (np.array([-0.11, -0.16]), np.array([0.11, 0.16]))
    x = np.array([0.30, 0.0])
    p, need = _friction_limited(np.array([-0.11, 0.0]), x, box, 0.25)
    assert need == pytest.approx(0.19) and np.linalg.norm(x - p) == pytest.approx(0.25)
    assert box[0][0] <= p[0] <= box[1][0]
    _, need = _friction_limited(np.array([-0.11, 0.0]), x, box, 0.1)
    assert need > 0.1  # no admissible CoP: a genuine slip


def test_design_levers_are_paired_and_signed():
    from breakpoint.design import LEVERS, Lever, lever_table, sign_test_p

    assert sign_test_p(0, 0) == 1.0 and sign_test_p(10, 0) == pytest.approx(2 / 2**10)
    assert sign_test_p(3, 3) == 1.0
    same = Lever("none", "No change", "Nobody")
    swing = next(lv for lv in LEVERS if lv.key == "swing_time")
    t = lever_table(trials=25, jobs=("trailer unloading", "home care"), levers=(same, swing))
    rows = {r["key"]: r for r in t["levers"]}
    assert rows["none"]["fixed"] == rows["none"]["broken"] == 0  # common random numbers: identical shifts
    assert rows["swing_time"]["broken"] == 0 and rows["swing_time"]["envelope"]["left"] > t["baseline"]["envelope"]["left"]
    for j in t["jobs"]:
        assert rows["none"]["jobs"][j]["failure_rate"] == t["baseline"]["jobs"][j]["failure_rate"]
