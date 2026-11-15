# Physical AI and robot morphologies: what Breakpoint models, and why

Short research notes behind the bodies and stresses in the humanoid lab. Every claim below is either
measured in this repository (with the command that reproduces it) or cited.

## Bodies

| Body | How it recovers from a shove | What stops it | In Breakpoint |
|---|---|---|---|
| Walking biped | Ankle torque, then capture-point stepping | Side shoves (the swing leg cannot cross the loaded one), slips, reach | `Humanoid` in `breakpoint/humanoid.py` |
| Wheeled base | Drives under its capture point | Any step taller than a threshold (2.5 cm) | `WheeledBase()` |
| Wheel-legged hybrid (four legs, a wheel at each foot) | Drives under its capture point; legs lift it onto steps | Steps over 15 cm | `WHEEL_LEGGED` |
| Quadruped (learned policy) | Whatever the PPO policy learned | Measured per axis by the breaking-point audit | `breakpoint/env.py`, MuJoCo |

The 15 cm step-up for the hybrid matches published learned stair climbing on a wheel-legged robot: an RL
policy climbed 15 cm steps blind, inferring the step from proprioception ([arXiv:2402.06143](https://arxiv.org/pdf/2402.06143)).
Balance-based controllers on 12-DoF wheel-legged robots have climbed far higher obstacles (0.36 m,
[arXiv:2109.09934](https://arxiv.org/pdf/2109.09934)), so 15 cm is a conservative, learned-controller figure.

Not modelled yet: a two-wheeled self-balancing body (it is a pendulum on wheels with no static footprint
fore and aft, and needs its own controller model).

**Mobile manipulation, first step.** A tote held further out shifts the combined centre of mass forward
(`python -m breakpoint humanoid --payload 20 --reach 0.45`, and the "Held this far out" slider). With
20 kg, moving it from 20 to 40 cm makes shoves from the front easier to survive and shoves from behind
harder (87 → 74 N·s). The model holds the body still rather than leaning back, so the slider stops at
40 cm (safe up to the 25 kg maximum), and the CLI refuses a load and reach that would put the standing
centre of mass past the foot.

## Stresses that come from the AI, not the floor

**Action chunking.** Vision-language-action models and diffusion or flow policies usually emit a chunk of
future actions and run it open-loop. That makes motion smooth, but a shove that lands mid-chunk is not
answered until the policy acts on a new observation. Real-time chunking generates the next chunk while the
current one runs and switches as soon as it is ready, which keeps the robot reactive under inference delay
([Black, Galliker and Levine, NeurIPS 2025, arXiv:2506.07339](https://arxiv.org/abs/2506.07339)).

`python -m breakpoint chunking` turns each scheme into a reaction delay and reads the cost off the push
envelope (biped, 50 ms inference). A synchronous loop observes, infers for 50 ms and runs the chunk, so a
shove at the worst moment waits a whole cycle plus inference. Read the table as *if the chunked policy
closes the balance loop itself*: real humanoid stacks usually keep a fast balance controller underneath,
which a chunk does not delay. It measures the first reaction only; afterwards the model re-plans every tick
(optimistic) and has no joint stiffness while it waits (pessimistic).

| Scheme | Wait for a shove at the worst moment | Front / side shove survived |
|---|---|---|
| No delay at all | 0 ms | 92 / 44 N·s |
| Synchronous, 100 ms chunks | 200 ms | 49 / 23 N·s |
| Synchronous, 400 ms chunks | 500 ms | 19 / 9 N·s |
| Real-time chunking | 100 ms | 67 / 32 N·s |

The wheel-legged hybrid loses far less (95 → 44 N·s from the front at 400 ms chunks), because its drive
can correct late without needing a foot placed in time. The model measures reaction delay only; it says
nothing about the smoothness at chunk boundaries that real-time chunking also fixes.

**Teleoperation lag** is the same mechanism with a human in the loop; `python -m breakpoint episodes`
tags recorded demonstrations whose lag or near-falls would teach a policy the wrong thing.

## Next

- A pendulum-on-wheels model for two-wheeled self-balancing robots.
- Arm motion during a task (a moving centre of mass), not just a static reach, and the lean-back a real
  controller uses to carry a load far out.
- Chunking with the policy's own inference-time distribution (measured, not a constant).
