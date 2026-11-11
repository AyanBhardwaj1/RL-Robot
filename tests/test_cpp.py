"""The C++ capture-point controller (cpp/) must agree with breakpoint/humanoid.py."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from breakpoint.humanoid import Conditions, envelope, recover

CPP = Path(__file__).resolve().parent.parent / "cpp"


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        pytest.fail("a C++17 compiler (g++ or clang++) is required for the C++ port tests")
    out = tmp_path_factory.mktemp("cpp") / "breakpoint_cp"
    subprocess.run([cxx, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-o", str(out),
                    str(CPP / "main.cpp")], check=True, capture_output=True, text=True, timeout=120)
    return out


def run(binary, *args):
    return json.loads(subprocess.run([str(binary), *map(str, args)], check=True, capture_output=True,
                                     text=True, timeout=120).stdout)


@pytest.mark.parametrize("imp,deg,lat,payload,mu", [
    (0, 0, 0, 0, 0.8), (40, 0, 0, 0, 0.8), (60, -90, 0, 0, 0.8), (95, 0, 0, 0, 0.8),
    (50, -90, 0.15, 0, 0.8), (30, 45, 0, 20, 0.8), (70, 0, 0, 0, 0.15),
    (60, 180, 0.025, 0, 0.8), (40, -135, 0.1, 10, 0.8), (45, 90, 0, 0, 0.8), (30, -90, 0, 0, 0.8), (60, 180, 0, 0, 0.1), (40, 180, 0, 0, 0.35), (55, 0, 0.05, 15, 0.45),
])
def test_cpp_recover_matches_python(binary, imp, deg, lat, payload, mu):
    got = run(binary, "recover", imp, deg, lat, payload, mu)
    py = recover(imp, deg, cond=Conditions(latency_s=lat, payload_kg=payload, friction=mu))
    assert got["recovered"] == py.recovered and got["steps"] == py.steps
    assert got["time_s"] == pytest.approx(py.time_s, abs=0.003)


def test_cpp_envelope_matches_python(binary):
    for lat in (0.0, 0.2):
        got = run(binary, "envelope", lat)
        py = envelope(cond=Conditions(latency_s=lat))
        assert got.keys() == py.keys()
        for k in py:
            assert got[k] == pytest.approx(py[k], abs=0.02), k


def test_cpp_realtime_step_is_fast(binary):
    b = run(binary, "bench", 0.3)
    assert b["ticks"] > 50
    assert b["step_us_p99"] < 200  # the control step itself, not the OS wake-up jitter


def test_cpp_negative_latency_is_treated_as_none(binary):
    got = run(binary, "recover", 40, 0, -0.5, 0, 0.8)
    py = recover(40, 0, cond=Conditions())
    assert got["recovered"] == py.recovered and got["steps"] == py.steps
