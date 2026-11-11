"""Example 'bring your own policy': a hand-written lateral-balance reflex.

    python -m breakpoint audit --policy py:examples.my_controller:make_policy

The observation layout is documented in breakpoint/env.py::_obs (gravity in body frame
first). This controller widens the stance (hip abduction) toward the side the body is
tipping to, and leaves sagittal motion to the CPG.
"""

import numpy as np


def make_policy():
    def act(obs: np.ndarray) -> np.ndarray:
        gy = float(obs[1])  # lateral component of gravity in the body frame (tilt)
        a = np.zeros(12)
        a[0::3] = np.clip(-2.5 * gy, -1, 1)  # abduction joints of all four legs
        return a

    return act
