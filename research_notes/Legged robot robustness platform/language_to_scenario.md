# Language/Vision-to-Scenario Generation for Robot RL (LLM/VLM-generated terrains, rewards, DR, test cases, 2023-2026)

Research note on how this was gathered: done under a 15-call limit. arxiv.org, eureka-research.github.io and par.nsf.gov could not be fetched in this sandbox (DNS blocked), so the findings come from search-result abstracts and summaries, not full texts. Exact tables (for example DrEureka's per-parameter ranges) could not be checked. Those are listed under Gaps.

## Q1. Key systems: what do they generate and how do they validate outputs?

### Takeaway
Most work in this area follows one pattern. The LLM writes executable code (reward functions, terrain or heightfield generators, task classes, scenario DSL scripts). The loop then filters that code by execution: compile/run checks, training rollouts, and fitness metrics. The results are fed back to the LLM for refinement. Validation through execution and policy performance is the main safeguard. Schema-only constraints are less common, though GenTe uses function calling over a fixed library of terrain primitives.

### Cited Findings
**Reward generation**
- Eureka (Ma et al., 2023): GPT-4 reads the raw environment source code plus a task description and writes reward code. Evolutionary search over reward-code candidates then refines it, using feedback from training. — [Eureka project page](https://eureka-research.github.io); [arXiv 2310.12931 summary](https://www.emergentmind.com/papers/2310.12931)
- The widely quoted claim that Eureka beats human-designed rewards on 83% of tasks with a 52% average normalized improvement comes from a secondary summary. The paper itself was not verified in this session. — [Liner review](https://liner.com/review/eureka-humanlevel-reward-design-via-coding-large-language-models)
- Text2Reward (Xie et al.): generates *dense*, interpretable reward code from goal text and supports refinement through human feedback. On ManiSkill2 and MetaWorld it matched the human-expert oracle on 13 of 17 tasks. It learned 6 novel locomotion behaviours with over 94% success. Human-feedback iterations raised some tasks from 0 to 1 success rate in the few-shot setting. — [Text2Reward arXiv 2309.11489](https://arxiv.org/html/2309.11489v3)
- A 2026 paper titled "When LLM Reward Design Fails" exists, which suggests a growing literature on failure modes of LLM reward design. Its content was not read. — [alphaxiv 2605.28918](https://www.alphaxiv.org/abs/2605.28918.md)

**Domain randomization (sim-to-real)**
- DrEureka (Ma et al., RSS 2024) has three stages:
  1. An LLM synthesizes the reward, with *safety instructions* added to the prompt.
  2. The initial policy is rolled out under perturbed physics to build a **Reward-Aware Physics Prior (RAPP)**. For each physics parameter, the RAPP range is the min and max value at which the policy still succeeded.
  3. The LLM receives RAPP and proposes several candidate DR distributions. Policies are retrained under each candidate and the best is selected.
  — [DrEureka arXiv 2406.01967 (via search)](https://arxiv.org/html/2406.01967v1); [project page](https://eureka-research.github.io/dr-eureka)

**Terrain and environment curricula (legged)**
- Eurekaverse (Liang et al., CoRL 2024 oral, UPenn): the LLM writes Python functions that output **height fields plus goal coordinates** for quadruped parkour. Generation alternates with training, which the authors call "agent-environment co-evolution". In later rounds the LLM is prompted with the performance metrics of the current policy and asked to vary environments that worked well. — [Eurekaverse arXiv 2411.01775](https://arxiv.org/pdf/2411.01775); [project page](https://eureka-research.github.io/eurekaverse)
- Eurekaverse validation: generated code goes through an "environment check". Just over 50% of GPT-4o terrains passed. Common errors were:
  - height values mis-scaled to the wrong units
  - out-of-bounds index access
  - references to helper functions or variables that do not exist

  The authors suggest that visual feedback during generation could improve spatial reasoning. — [Eurekaverse paper](https://arxiv.org/pdf/2411.01775) (via search summary)
- Eurekaverse results: it beat human-designed training courses in the real world and outperformed baselines and ablations on 20 held-out simulated parkour courses. It handled large gaps, steep ramps and a deformable yoga ball. — [Eurekaverse arXiv](https://arxiv.org/pdf/2411.01775); [CoRL listing](https://mlanthology.org/corl/2024/liang2024corl-environment)
- GenTe (arXiv 2504.09997, Apr 2025): a VLM with **function calling** turns text *and images* into terrains by composing an **atomic terrain library**. The library covers geometric terrain (height maps, obstacles) and physical terrain, with force models for soil sinkage and hydrodynamic resistance. It ships a benchmark of 100 generated terrains, reports better generalization for bipedal locomotion, and the code is at github.com/HaronW/GenTe. — [GenTe arXiv](https://arxiv.org/html/2504.09997v1)
- CurricuLLM (Ryu et al., ICRA 2025, Berkeley) has three steps:
  1. The LLM writes a sequence of subtasks in natural language.
  2. It translates each subtask into executable reward code and goal-distribution code.
  3. Trained policies are evaluated from trajectory rollouts against the subtask description.

  It was tested on manipulation, navigation and locomotion, and transferred to the real Berkeley Humanoid. Code is at github.com/labicon/CurricuLLM. — [CurricuLLM arXiv 2409.18382](https://arxiv.org/html/2409.18382v2); [ICRA PDF](https://hybrid-robotics.berkeley.edu/publications/ICRA2025_CurricuLLM.pdf)
- AURA (arXiv 2506.02507, 2025): an agentic, retrieval-augmented framework that goes from a natural-language prompt to a curriculum to a controller. Policies were deployed zero-shot on a custom kid-sized humanoid, and it reportedly outperformed LLM-guided baselines. — [AURA arXiv](https://arxiv.org/html/2506.02507v3)

**Task and scene generation (manipulation and embodied AI)**
- GenSim (Wang et al., ICLR 2024): uses LLM program synthesis to generate simulation *tasks*. It has two modes: goal-directed (a curriculum toward a target task) and exploratory (new tasks proposed from earlier ones). A **task library** stores validated high-quality tasks as few-shot references. Validation combines code reflection with human verification, described in the appendix. — [GenSim arXiv 2310.01361](https://arxiv.org/pdf/2310.01361); [ICLR poster](https://iclr.cc/virtual/2024/poster/18747)
- GRS (arXiv 2410.15536): generates sim tasks from real RGB-D images. It writes a **tailored test suite for each simulation** to check fidelity to the task description. An LLM "router" reads runtime errors and test results to refine the code. — [GRS arXiv](https://arxiv.org/html/2410.15536v1)
- RoboGen: runs a generative-simulation "propose-generate-learn" cycle in which foundation models create tasks, scenes and training supervision. — [Stanford CS224N report citing RoboGen](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/AngelaYiRanajitGangopadhyayYihanZhou.pdf)
- Holodeck: generates 3D embodied-AI environments from language by breaking scene generation into structured steps. — [same CS224N report](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/AngelaYiRanajitGangopadhyayYihanZhou.pdf)

**Real-to-sim from images**
- URDFormer (UW/NVIDIA): a transformer maps a single RGB image to a URDF scene with articulated kinematic structure. It is suited to pretraining on many approximate scenes and is less accurate than RialTo. — [URDFormer arXiv 2405.11656](https://arxiv.org/html/2405.11656v1); [TechXplore](https://techxplore.com/news/2024-08-photos-videos-ai-conjure-simulations.pdf)
- RialTo (RSS 2024): builds digital twins, including articulated objects, through a scanning interface. Policy robustness improved by over 67% in real tests. — [RSS 2024 program](https://roboticsconference.org/2024/program/papers/15/)
- "Splatting Physical Scenes" (arXiv 2506.04120, 2025): a hybrid of 3D Gaussian splats for appearance and explicit meshes for physics. It jointly optimizes mesh geometry, camera poses and robot joint angles through differentiable rendering and **differentiable MuJoCo** physics, using raw ALOHA 2 data. — [arXiv 2506.04120](https://arxiv.org/abs/2506.04120v2)
- 3DGS digital twin (arXiv 2601.03200, 2026): lifts **SAM** 2D masks into the 3D Gaussian scene using multi-view consensus, then converts splats into collision geometry. The simulator is Unity, not MuJoCo. — [arXiv 2601.03200](https://arxiv.org/html/2601.03200v1)
- GSWorld (arXiv 2510.20813, 2025): reconstructs a metric-scale twin from multi-view capture, sets scale with ArUco markers, aligns the URDF by ICP, uses a Gaussian-on-Mesh (GSDF) asset format and supports parallel visual RL. — [arXiv 2510.20813](https://arxiv.org/pdf/2510.20813v1)

### Inferences
- **For a small MuJoCo legged project, Eurekaverse-style heightfield generation is the closest analogue.** MuJoCo supports `hfield` assets natively, so an LLM or a template can emit a 2D numpy array plus metadata. GenTe's design is safer than free-form code: the LLM *selects and parameterizes* primitives from a fixed library, such as stairs(step_h, n), gap(width), slope(deg) and rough(amp, freq), instead of writing arbitrary code.
- Image-to-terrain is feasible in a lightweight form. A VLM (or an offline heuristic) maps a photo to a terrain *class plus parameters* (for example "gravel slope ~15°"), which then feeds the same primitive library. A full Gaussian-splat or SAM reconstruction is heavy and mainly targets manipulation scenes.
- Holodeck (from background knowledge, not verified in this session) has the LLM output spatial-relation constraints in a structured format, which a classical constraint solver then places. This "LLM proposes relations, solver guarantees validity" split carries over well.

### Gaps
- Full texts of RoboGen and Holodeck validation procedures could not be fetched. Holodeck's constraint-solver detail and RoboGen's use of LLM-chosen physical parameters (mass, friction) come from prior knowledge and are unverified here.
- Eureka's 83%/52% figures were seen only in a secondary source.
- No published end-to-end "image → SAM → MuJoCo terrain → RL" pipeline for legged robots was found.

## Q2. How do these systems constrain LLM outputs to physically valid parameters?

### Takeaway
Five constraint mechanisms recur:
- (a) constrained output spaces, such as function calling over a primitive library (GenTe) or snippet retrieval for a DSL (ChatScene)
- (b) empirically derived bounds that the LLM must sample within (DrEureka's RAPP)
- (c) execution and environment checks that reject broken code (Eurekaverse, GRS)
- (d) generated unit tests plus error-routing critique loops (GRS, GenSim reflection)
- (e) selection by downstream RL performance (Eureka's evolutionary search, Eurekaverse co-evolution, DrEureka candidate selection)

Free-form LLM code without these checks fails often: about half of Eurekaverse terrains failed the check.

### Cited Findings
- DrEureka bounds LLM DR proposals with RAPP, the per-parameter [min, max] range in which the initial policy still succeeds in sim. The LLM only proposes distributions inside these grounded ranges. — [DrEureka arXiv](https://arxiv.org/html/2406.01967v1)
- Eurekaverse uses an environment check before training. About 50% of GPT-4o terrains failed it, mainly from unit/scale errors, out-of-bounds indexing and hallucinated helpers. — [Eurekaverse](https://arxiv.org/pdf/2411.01775)
- ChatScene (CVPR 2024): prompting the LLM to write Scenic DSL directly produced non-executable code and calls to Scenic APIs that do not exist. The fix was to have the LLM first build a **retrieval database of code snippets**, organized by the 4-part script structure: map/model settings, adversarial behaviours and so on. Scripts are then assembled from retrieved, known-good snippets. — [ChatScene arXiv 2405.14062](https://arxiv.org/html/2405.14062v1)
- One 2026 student report describes ChatScene as *fine-tuning* an LLM. That conflicts with the original paper, which uses retrieval. The same report proposes a JSON-based scenario representation as an alternative to DSL code. — [UB tech report 2026-22](https://cse.buffalo.edu/tech-reports/2026-22.pdf); contradicted by [ChatScene paper](https://arxiv.org/html/2405.14062v1)
- GRS writes a test suite per generated simulation, and an LLM router uses runtime errors and test outcomes to drive repair. — [GRS](https://arxiv.org/html/2410.15536v1)
- GenTe constrains output through function calling over an atomic terrain library. — [GenTe](https://arxiv.org/html/2504.09997v1)
- GenSim keeps a curated task library of verified tasks and uses it as few-shot context, so new generations stay close to known-valid code. — [GenSim](https://arxiv.org/pdf/2310.01361)
- CurricuLLM closes the loop by evaluating rollouts against each subtask's natural-language description. — [CurricuLLM](https://arxiv.org/html/2409.18382v2)

### Inferences (implementation recipe for Python + MuJoCo, optional LLM, offline fallback)
1. **Schema first.** Define a Pydantic/JSON-Schema `Scenario` with these fields:
   - `terrain`: a list of `{primitive: enum, params}`
   - `physics`: friction, mass scale, motor strength, gravity and so on, each with hard `min`/`max`
   - `disturbances`: push force, timing, duration
   - `sensor_noise`
   - `success_criteria`

   Use the LLM provider's structured-output or tool-calling mode and reject anything that fails validation (the GenTe and ChatScene lessons).
2. **Hard clamps plus physical sanity checks.** Before running, clamp every value to engineering bounds. Then check that the MuJoCo model compiles, that the robot spawns without interpenetration, and that a no-op or default policy runs N steps without NaNs. This is the Eurekaverse environment check.
3. **Grounded bounds (cheap RAPP).** Sweep each parameter one at a time with the current policy and record the range in which it still succeeds. Give these ranges to the LLM, or to an offline sampler, as allowed bounds (DrEureka).
4. **Critique/repair loop.** On validation or runtime failure, send the error back to the LLM up to k times (GRS), then fall back.
5. **Offline fallback.** Deterministic keyword or regex parsing maps text to primitives (for example "stairs" → stairs primitive, "icy" → friction 0.1-0.3) and fills defaults from preset templates. The same schema and validator are used either way, so downstream code does not know whether an LLM was involved.
6. **Library of accepted scenarios** (GenSim) that serves as few-shot examples and as a regression test suite.

### Gaps
- No source quantified how much structured outputs (JSON schema) reduce invalid-scenario rates compared with free code for robot sim specifically. The ChatScene and Eurekaverse failure modes suggest the gain is large, but no number was found.

## Q3. What did DrEureka find about LLM-proposed DR ranges versus human-tuned ones?

### Takeaway
On a Unitree Go1 quadruped, DrEureka's LLM-generated reward plus DR beat the human-designed setup by 34% in forward velocity and 20% in distance travelled across real terrains. It also beat human design on LEAP-hand cube rotation and enabled walking on a yoga ball. Two ablations matter: the reward-aware physics prior is crucial, and LLM-proposed distributions within RAPP beat random sampling within the same RAPP ranges. This indicates that the LLM adds useful priors beyond the bounds themselves.

### Cited Findings
- DrEureka policies outperformed human-designed reward and DR by **34% forward velocity** and **20% distance travelled** across various real-world terrains for quadruped locomotion. — [DrEureka arXiv 2406.01967](https://arxiv.org/html/2406.01967v1); [project page](https://eureka-research.github.io/dr-eureka)
- On LEAP-hand in-hand cube rotation, DrEureka beat the manually designed approach on number of rotations. — [DrEureka arXiv](https://arxiv.org/html/2406.01967v1)
- Ablation: using the initial policy to build RAPP is "crucial". Sampling DR uniformly at random within RAPP ranges does worse than LLM-proposed distributions. — [DrEureka arXiv](https://arxiv.org/html/2406.01967v1) (via search summary); [Moonlight review](https://www.themoonlight.io/review/dreureka-language-model-guided-sim-to-real-transfer)
- Safety instructions in the reward prompt were described as critical for producing rewards safe enough to deploy on real hardware. — [DrEureka arXiv](https://arxiv.org/html/2406.01967v1)
- Limitation noted by a secondary summary: DR parameters are static and not adapted during training. — [Medium summary](https://medium.com/antaeus-ar/dreureka-revolutionizing-robot-training-with-llms-331f2742c725) (secondary, lower confidence)

### Inferences
- For a robustness platform, the transferable lesson is to **ground before generating**. Empirical feasibility sweeps (RAPP) bound the LLM. The LLM's role is to choose *which* sub-ranges and parameters matter for the task, for example wider friction for a slippery deployment or narrower gravity. An offline fallback could reasonably use "uniform within RAPP", which DrEureka reports as somewhat worse but still a sensible baseline.

### Gaps
- The exact parameter list was not verified. From background knowledge it likely included friction, restitution, payload mass, CoM displacement, motor strength, PD gains, gravity, and possibly push and terrain parameters. Also unverified: the number of LLM candidates and how the LLM ranges compared with human ranges (wider or narrower). The full paper (arxiv.org, par.nsf.gov) was unreachable from this sandbox. The report writer should not state specific per-parameter ranges.

## Q4. Work on LLM/VLM agents as adversaries or red-teamers for robot policies

### Takeaway
LLM red-teaming of robot policies exists but mostly targets *instructions* for language-conditioned policies (ERT, Q-DIG), or comes from autonomous driving (ChatScene: LLM writes adversarial-vehicle scenarios, and fine-tuning on them cuts collisions). Non-LLM geometric red-teaming (GRT) finds physical failure shapes. No paper was found in which an LLM adversarially generates *terrain/physics/disturbance* scenarios to break a legged locomotion policy. That looks like an open niche.

### Cited Findings
- Embodied Red Teaming (ERT, arXiv 2411.18676, v2 Feb 2025): VLM-based automated red-teaming generates contextually grounded, hard instructions. State-of-the-art language-conditioned robot models fail or act unsafely on them. — [arXiv 2411.18676](https://arxiv.org/abs/2411.18676v2)
- Q-DIG (arXiv 2603.12510, Mar 2026): combines Quality-Diversity optimization with VLMs to generate a diverse range of adversarial instructions that expose VLA vulnerabilities. It names controllable diversity of failures as the open challenge. — [arXiv 2603.12510](https://arxiv.org/html/2603.12510)
- Geometric Red-Teaming (GRT, CoRL 2025 oral, not LLM-based): generates "CrashShapes", mesh deformations that break pretrained manipulation policies. In real-arm tests, task success dropped from 90% to as low as 22.5%. — [arXiv 2509.12379](https://arxiv.org/html/2509.12379v1)
- ChatScene (CVPR 2024): LLM agent turns text descriptions of dangerous driving situations into CARLA/Scenic scenarios with adversarial surrounding vehicles. Fine-tuning RL driving agents on them gave about a 9% reduction in collision rate. — [ChatScene](https://arxiv.org/html/2405.14062v1); code at github.com/javyduck/ChatScene

### Inferences
- An "LLM red-team" for a legged robot can follow a ChatScene/Q-DIG loop:
  1. The LLM proposes a scenario in the validated schema, for example "push 80 N lateral at mid-stride on a 10° icy slope".
  2. The sim runs it and returns structured failure metrics: fell, time-to-fall, max tilt, slip.
  3. The LLM proposes harder or more diverse variants.
  4. Failures are archived in a QD-style grid indexed by (terrain type, disturbance type) to keep diversity.

  The offline fallback is random or CMA-ES search over the same schema. Found failures then become a curriculum or regression test set (ChatScene shows retraining on them helps).

### Gaps
- No LLM-driven adversarial *physical-scenario* generation for legged locomotion was found within the search budget. 2025-2026 work in this space may exist but was not surfaced.
