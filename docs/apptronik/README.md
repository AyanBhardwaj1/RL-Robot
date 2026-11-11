# Breakpoint × Apptronik: application pack

This folder is for my software engineering internship application to Apptronik. A one-page brief to hand
over or attach is in [`breakpoint-one-pager.pdf`](breakpoint-one-pager.pdf) (source: `one-pager.html`). It holds four things:

- a research brief on the company and its roles (sources linked),
- a map from each team and intern posting to what Breakpoint already shows,
- a 5-minute demo script,
- interview prep.

Research date: 11 Nov 2026. The research came from search summaries of public pages: the company site and the
Greenhouse board could not be opened from the build environment, so **re-check every fact against the
originals before quoting it**. Items marked *(unverified)* came from a single aggregator.

---

## 1. The company in one page

- **Who:** Apptronik, Austin, Texas. Spun out in 2016 from Luis Sentis’s Human Centered Robotics Lab at UT
  Austin. Co-founders include Jeff Cardenas (CEO) and Nick Paine (CTO). Paine’s PhD was on high-performance
  series elastic actuation, and he and Sentis worked on NASA’s Valkyrie for the DARPA Robotics Challenge.
  ([NASA spinoff](https://spinoff.nasa.gov/node/11613),
  [HCRL Valkyrie actuator paper](https://sites.utexas.edu/hcrl/wp-content/uploads/sites/3888/2016/01/jfr-valkyrie-actuator-control-final.pdf))
- **Robots before Apollo:**
  - Draco, a biped with liquid-cooled viscoelastic actuators ([arXiv 1901.08100](https://arxiv.org/abs/1901.08100))
  - Astra, a force-controlled upper body
  - QDH, the Apollo prototype ([IEEE Spectrum](https://spectrum.ieee.org/humanoid-robot-apptronik-apollo))
- **Apollo:**
  - Unveiled in Aug 2023: 1.73 m, about 73 kg, 25 kg payload, swappable batteries of about 4 h each. It comes as a biped, on a wheeled base, or on a stationary mount.
  - Uses custom linear and rotary electric actuators. Actuators and drives are over half the bill of materials.
  - Sources: [New Atlas](https://newatlas.com/robotics/apptronik-apollo-humanoid-gpr/), [Engadget](https://www.engadget.com/apptroniks-apollo-is-the-latest-humanoid-robot-to-beat-tesla-to-market-164246161.html)
- **Apollo 2** (30 Jun 2026):
  - Comes as a biped or on a wheeled base “built to meet existing industrial safety standards”.
  - Positioned for data collection and scaled pilots.
  - Apollo 3 is expected to be the first true product (2027).
  - Sources: [Robotics 24/7](https://www.robotics247.com/article/apptronik-unveils-apollo-2-humanoid-robot-opens-robot-park-data-collection-and-training-facility), [Forbes](https://www.forbes.com/sites/johnkoetsier/2026/06/30/apptronik-announces-robot-park-a-90000-square-foot-humanoid-data-factory-teases-new-robot/)
- **Robot Park:** a roughly 90,000 sq ft site in Austin that collects teleoperation and autonomy data feeding Google DeepMind’s Gemini Robotics models ([GlobeNewswire](https://www.globenewswire.com/news-release/2026/06/30/3319598/0/en/Welcome-to-Robot-Park-Where-Apptronik-s-Apollo-Goes-to-Work-Training-the-Next-Generation-of-Humanoid-Robot-Intelligence.html)).
- **Partners:**
  - Mercedes-Benz: line-side parts delivery and inspection pilots.
  - GXO: logistics proof of concept.
  - Jabil: manufacturing pilots, and Jabil will build Apollo at scale.
  - Google DeepMind: models (Gemini Robotics 1.5 demos ran on Apollo).
  - NVIDIA: GR00T and Isaac.
- **Funding:**
  - Series A of $350M (Feb 2025), extended by $53M (Mar 2025) and by $520M (Feb 2026), for about $935M in the round.
  - Valuation reported at about $5B or more.
  - Source: [TechCrunch](https://techcrunch.com/2026/02/11/humanoid-robot-startup-apptronik-has-now-raised-935m-at-a-5b-valuation/)
- **Status:** customer work is still pilots, and no public uptime or task-throughput numbers exist.
- **Values** ([careers/culture](https://apptronik.com/careers/culture)): Creativity, Humility, Integrity, Passion, Ask Why.

## 2. What they hire for

| Team | What the postings ask for | Where Breakpoint shows it |
|---|---|---|
| Controls / motion control and planning | Whole-body control (OSC, QP, MPC), state estimation, real-time C++ and Python, RL in sim deployed to hardware | `breakpoint/humanoid.py` (capture-point push recovery), `/humanoid` lab |
| Simulation | MuJoCo (the most-cited simulator), Isaac Sim/Lab, contact models, domain randomization, MJCF/URDF | `world.py`, `env.py`, the scenario schema |
| Robot learning / foundation models | RL, imitation learning, VLAs, transformers | numpy PPO, hardening loop, honest training results |
| ML systems / ML Ops | Profiling, quantizing, distilling RL and VLA policies for onboard compute; ONNX, TensorRT | `quantize.py`: INT8 is 3.6× smaller with no readiness loss |
| Perception | Detection, SLAM, sensor fusion, ROS 2, edge GPUs | photo-to-scenario, sensor-noise axis |
| Embedded / firmware | Motor control (FOC), EtherCAT, CAN, embedded Linux | latency and actuator-strength axes; C++17 controller and real-time loop benchmark |
| Platform / dev tools | Bazel, CI/CD, developer experience, AI coding agents | CI, over 90 tests, three implementations (Python, JS, C++) held equal by tests, CLI |
| Fleet Connect / integration | gRPC, protobuf, REST, MQTT, Kafka, WMS/ERP integration | stateless HTTP API, deployment profiles |
| Test / TestOps | HIL and simulation testing, regression, coverage | `regress`, `certify`, shrunk counterexamples |
| Safety | IEC 61508, ISO 13849, ISO 10218, R15.06/R15.08, HARA, FMEA | failure bounds, evidence-based explanations, biped vs wheeled odds |

Role sources: [Staff SWE – Humanoid Controls](https://job-boards.greenhouse.io/apptronik/jobs/6142283004),
[Perception SLAM](https://job-boards.greenhouse.io/apptronik/jobs/6013277004),
[Simulation Engineer](https://builtin.com/job/simulation-engineer-design-and-behavior-dexterity/8520184),
[Fleet Connect](https://builtin.com/job/full-stack-software-engineer-fleet-connect/9130239),
[Product Safety](https://builtin.com/job/staff-systems-engineer-product-safety/8281178).
A third-party count put open roles at about 86 in Oct 2026 *(unverified)*.

## 3. Intern postings and what to show for each

| Intern posting (seen 2023–2026) | What it asks for | Show them |
|---|---|---|
| **SWE Intern – Developer Experience** ([Built In](https://builtin.com/job/software-engineering-intern-developer-experience/8766589)) | C++, Python or TypeScript; CI/CD, Docker; using coding agents | CLI, CI, tests, the serverless site, and how an AI agent was used and then checked by reviews |
| **SWE Intern – ML Systems / ML Ops** ([Greenhouse](https://job-boards.greenhouse.io/apptronik/jobs/6128057004)) | Profile, quantize and distill RL/VLA policies; PyTorch, ONNX, TensorRT | `python -m breakpoint quantize`: compression measured by robustness, not only by action error |
| **Robotics Software Intern – Real-Time Controls** ([Greenhouse](https://job-boards.greenhouse.io/apptronik/jobs/5985132004)) | Real-time C++, unit tests, actuators, encoders, IMUs | `cpp/capture_point.hpp`: an allocation-free 2 ms controller step, parity tests against Python, a 500 Hz benchmark (0.8 µs step, 4.9 ms worst wake-up jitter on a stock kernel) |
| **Autonomy Software Intern** ([Built In](https://builtin.com/job/autonomy-software-intern-humanoid-robotics/8601758)) | Behaviour execution in sim and on hardware; ROS 2, MuJoCo, Pinocchio | scenario-driven evaluation, job profiles |
| **Motion Control and Planning Intern** ([Built In](https://www.builtinaustin.com/job/motion-control-and-planning-intern/8816589)) | RL, robot dynamics, PyTorch or JAX, MuJoCo or Drake | LIP and capture point, PPO from scratch |
| **Perception Software Intern** ([Built In](https://builtin.com/job/perception-software-intern-humanoid-robotics/8376319)) | OpenCV, ROS 2, 3D vision | photo-to-scenario |
| **SWE Test Ops Intern** ([b.capital](https://jobs.b.capital/companies/apptronik/jobs/41789961-software-engineer-test-ops-intern-2025)) | HIL and simulation tests, CMake or Bazel, Python | `regress`, `certify`, the hosted-API tests |

Shared requirements: internships run 12 weeks (summer or fall), are based in Austin, and some
postings ask for a 3.0 GPA. Interview reports *(thin data)* describe a recruiter screen, a “tech bio”, a
deep dive on past robotics projects, and 4–5 technical rounds for full-time roles
([Glassdoor](https://www.glassdoor.com/Interview/Apptronik-Interview-Questions-E1750713.htm)).

## 4. The 5-minute demo

1. **Hook (30 s).** Open `/humanoid` and click beside the robot. It takes a step and the capture point
   settles inside the new foot. Shove harder from the rear and it falls after 3 steps. Say: “Every
   number here comes from the linear inverted pendulum and capture-point theory, with a real robot’s
   limits.”
2. **The product question (60 s).**
   - Shove from the side: the biped holds only 44 N·s, half its front limit, because the stepping leg
     cannot cross the loaded one. Switch to the wheeled base: 120 N·s from the side, 137 from the front.
   - Raise the curb slider to 10 cm and it is blocked.
   - Scroll to “Legs or wheels, by job”. Home care: wheeled fails 47%, biped 0%. Trailer
     unloading: wheeled fails 16% (blocked), biped 13% (mostly slips).
   - Show the capturability map, then drag latency to 150 ms. Say: “This map caught a bug in my own
     latency model that bisection missed, and an independent review caught two more. Each fix made the
     biped look worse and the numbers more honest.”
   - Say: “This is the biped-versus-wheeled trade-off you shipped in Apollo 2, made measurable.”
3. **Readiness, honestly (90 s).** Open the quadruped report.
   - The first trained policy looked fine in training but scored below the gait it was built on.
   - Hardening fixed it; a second round overfit; latency could not be trained away.
   - A latency gate took teleop failures from 10% to 0%.
   - Say: “The tool caught my own mistakes.”
4. **Ship it (60 s).** Run `python -m breakpoint quantize --policy runs/v2field/policy_v2field.npz --gated`: INT8 is 3.6× smaller with the same readiness score. Then run `certify`: a certificate with hash and versions that fails CI below a bar.
5. **Real time (optional, 30 s).** Run `./build-cpp/breakpoint_cp bench 2`. The step takes about 1 µs; the
   scheduler's wake-up jitter reaches about 5 ms. Say: “The math is free; the kernel is the risk.”
6. **Close (30 s).** Open `/roles`: each Apptronik team, the file that touches it, and what I’d build next with them.

## 5. Talking points and honest limits

- **Physics:** the humanoid is a reduced model (an inverted pendulum with massless legs). It ignores angular
  momentum and swing-leg dynamics, does not alternate left and right feet, and lets any foot step
  anywhere within reach (except crossing the stance leg sideways). Next: check it against a full MuJoCo
  humanoid under a QP whole-body controller.
- **Measurement noise:** breaking points come from bisection, so tiny policy changes move them. The FP16
  run moved the score by 0.7. Report scores with their spread.
- **Not a model of Apollo:** the 72 kg, 0.95 m robot is generic. Its numbers compare designs; they do not
  certify a product.
- **Process:** I built this with an AI coding agent, which the DevEx intern posting asks about. Two
  independent code reviews found 17 issues. The worst was mirrored push directions, caught after the
  results were published; I fixed and migrated them, and a physical regression test now pins the
  convention.

## 6. Questions to ask them

- How do you decide when a pilot task is ready to run unsupervised? Is there an internal readiness metric?
- Where does the biped beat the wheeled Apollo 2 in your customers’ sites today, and where is it the reverse?
- How do Robot Park teleoperation logs get filtered before they train Gemini Robotics policies?
- What does the sim-to-real gap look like for your actuators: friction, backlash, thermal limits?
- How is a compressed policy validated before it goes on the robot?

## 7. A short outreach note (edit before sending)

> Hi [name], I’m applying for Apptronik’s software engineering internship ([team]). I built Breakpoint,
> an open-source robustness lab for legged-robot policies. It finds the exact shove, slope, load and lag
> where a policy fails, hardens the policy, and gives deployment odds with confidence bounds. For Apollo’s world I
> added a humanoid push-recovery lab using capture-point stepping, which compares a walking biped with a
> wheeled base by job: legs win home-care routes 0% to 47%, wheels survive 2.7× bigger side shoves.
> I also measured that an INT8 policy is 3.6× smaller with no robustness loss. Demo: [link]. Code:
> https://github.com/AyanBhardwaj1/RL-Robot. I’d love to bring this to [team]; happy to walk through it in five minutes.
