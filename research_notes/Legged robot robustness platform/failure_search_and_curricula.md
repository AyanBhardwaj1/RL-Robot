# Adversarial Failure-Mode Search and Stress Testing for Learned Legged Locomotion Policies (2023-2026)

Method note: arxiv.org and roboticsproceedings.org could not be fetched in this session (DNS blocked by the sandbox network policy), so findings below come from search-result abstracts, project/aggregator summaries and proceedings listings rather than full-text reading. Quantitative claims are only included where a snippet stated them; everything else is flagged in Gaps. Papers before 2023 are marked [FOUNDATIONAL].

## Q1. What search methods are used to find failures, and which are most sample-efficient?

### Takeaway
Four families dominate: (a) learned RL adversaries (RARL lineage → H-infinity disturber 2024, Shi et al. RSS 2024 sequential attacks, PL-CAP humanoid 2025), (b) black-box optimizers over a low-dimensional disturbance/scenario parameter vector (CEM, differential evolution, random search, Bayesian optimization incl. multi-fidelity BO), (c) adaptive stress testing (AST; MCTS/RL over disturbance sequences to find the *most likely* failure), and (d) quality-diversity (MAP-Elites) archives that cover *many distinct* failure modes (e.g., QD terrain generation for Cassie/ANYmal, 2025). For a small CPU project, black-box search (CEM or CMA-ES) over a ~3-10D disturbance vector is the cheapest to implement and run, while QD gives the best coverage of failure *types*. Learned adversaries find stronger time-varying attacks but cost a full RL training run.

### Cited Findings
**Learned RL adversaries (state-dependent, sequential)**
- [FOUNDATIONAL] RARL (Pinto et al., ICML 2017) trains the agent together with an adversary that applies disturbance forces, in a zero-sum minimax game. The adversary's policy is itself trained by RL, so it learns how to destabilize the agent. It was tested on InvertedPendulum, HalfCheetah, Swimmer, Hopper, Walker2d (and Ant in the ICML version). The authors report better training stability, robustness to train/test mismatch, and beating the baseline even when no adversary is present at test time — [PMLR](https://proceedings.mlr.press/v70/pinto17a.html); [CMU RI](https://www.ri.cmu.edu/publications/robust-adversarial-reinforcement-learning)
- Shi et al., "Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers" (RSS 2024, arXiv 2405.12424). RL-trained adversaries produce *low-magnitude sequential* perturbations aimed at observations, commands and external perturbations. Even state-of-the-art robust controllers can fail significantly under well-designed low-magnitude sequences. The work was validated in simulation and on a real robot — [arXiv abs](https://arxiv.org/abs/2405.12424); [RSS XX proceedings](https://www.roboticsproceedings.org/rss20/p057.html); [aitopics summary](https://aitopics.org/doc/arxivorg:C30F2B54)
- Long et al., "Learning H-Infinity Locomotion Control" (CoRL 2024, arXiv 2404.14405). A learnable disturber chooses external forces based on the robot's state and is trained adversarially against the policy. An H-infinity constraint bounds the ratio of the disturber's cost to the force it applies, which keeps the joint optimization stable. The authors argue that randomly sampled external forces are "not efficient enough" to generate high-quality disturbance-resisting samples. The policy was deployed on real Unitree robots, including hind-legs-only standing — [alphaxiv](https://alphaxiv.org/abs/2404.14405); [mlanthology CoRL 2024](https://mlanthology.org/corl/2024/long2024corl-learning); [project PDF](https://junfeng-long.github.io/HINF/static/pdfs/hinf.pdf)
- Zhang et al. (SJTU), "Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots" (arXiv 2507.08303, Jul 2025; later retitled "Keep on Going: Learning Robust Humanoid Motion Skills via Selective Adversarial Training"). A learnable attack network finds weak points in the policy. The attacker's cost is constrained, so the attacks are "critical" (they improve robustness instead of just wrecking performance). Tested on Unitree G1 for perceptive locomotion and whole-body tracking. The attacker is used only during training — [arXiv html v1](https://arxiv.org/html/2507.08303v1); [emergentmind](https://www.emergentmind.com/papers/2507.08303)

**Black-box optimization over disturbance parameters**
- [FOUNDATIONAL, 2022] "Adversarial joint attacks on legged robots" (arXiv 2205.10098). Perturbs joint torque commands to find instabilities in walking. Compares black-box searches including random search and differential evolution, and is independent of the policy's architecture — [arXiv PDF](https://arxiv.org/pdf/2205.10098.pdf); [deeppaper](https://arxiv.deeppaper.ai/papers/2205.10098v1)
- [FOUNDATIONAL, 2012] CEM for falsification. Sankaranarayanan & Fainekos used CEM to adaptively choose input distributions for falsifying temporal-logic properties of hybrid systems. CEM guided by trace robustness found violations faster and more consistently than competing methods on their benchmarks — [ASU record](https://asu.elsevierpure.com/en/publications/falsification-of-temporal-properties-of-hybrid-systems-using-the-/)
- Multi-fidelity BO for falsifying learning-based controllers (Shahrooei, Kochenderfer, Baheri, arXiv 2409.08097, 2024). Chooses both the test input and the simulator fidelity. Reported as more computationally efficient than full-fidelity BO and other baselines for finding counterexamples on Gym environments — [arXiv PDF](https://www.arxiv.org/pdf/2409.08097)
- Ramezani & Åkesson, "Falsification of CPS using Bayesian Optimization" (ACM TECS 2025; preprint 2022). The choice of acquisition function changes how many simulations falsification needs. Vanilla BO is not well suited to high-dimensional falsification, which motivates TuRBO and πBO — [fugumt mirror](https://fugumt.com/fugumt/paper_check/2209.06735v3_enmode); [Chalmers](https://research.chalmers.se/publication/545697/file/545697_Fulltext.pdf)
- Moss, Kochenderfer et al., "Bayesian Safety Validation" (arXiv 2305.02449, 2023). Treats black-box safety validation as BO aimed at estimating failure probability. Falsification and most-likely-failure analysis come out as by-products — [arXiv PDF](https://pdf.arxiv.org/pdf/2305.02449)
- Dawson & Fan, "A Bayesian approach to breaking things" (CoRL 2023, PMLR 229). Frames failure-mode discovery as approximate Bayesian inference, using differentiable simulation. Predicts a "more diverse, representative set of failure modes than optimization-based falsification" and needs "up to 2x fewer iterations to converge relative to gradient-free techniques". It also repairs the failures it finds — [PMLR](https://proceedings.mlr.press/v229/dawson23a.html)
- Diffusion-based failure sampling (DiFS, arXiv 2406.14761, 2024). Claims better sample efficiency and mode coverage than existing black-box methods, and describes CEM as a parametric method that iteratively updates toward the failure distribution — [ar5iv](https://ar5iv.labs.arxiv.org/html/2406.14761)
- [FOUNDATIONAL, 2018] "Rigorous Agent Evaluation" (arXiv 1812.01647, DeepMind). Learns a failure-probability predictor over initial conditions and uses it to focus testing. Reports finding humanoid-locomotion failures with about 3,100x fewer samples than random testing — [arXiv html](https://arxiv.org/html/1812.01647); [bytez](https://bytez.com/docs/arxiv/1812.01647/paper)

**Adaptive stress testing (AST)**
- AST is an RL/MCTS framework that searches over disturbance sequences for the most likely path to failure in simulation (Lee et al., JAIR 2020) — [JAIR](https://jair.org/index.php/jair/article/download/12190/26631)
- In a flight-management case study, AST found more failures, and more likely ones, than direct Monte Carlo and a CEM importance-sampling baseline — [ar5iv 2011.02559](https://ar5iv.labs.arxiv.org/html/2011.02559)
- Survey of black-box safety validation algorithms (Corso et al., arXiv 2005.02979) groups methods into optimization, path planning, importance sampling and RL — [arXiv PDF](https://arxiv.org/pdf/2005.02979) (only the reference list was visible in search, so the survey's own conclusions are unverified)

**Quality-diversity / MAP-Elites**
- Esquerre-Pourtère, Kim, Park, "Generating Diverse Challenging Terrains for Legged Robots Using Quality-Diversity Algorithm" (arXiv 2506.01362, Jun 2025). Builds a QD archive of terrains in which each cell holds the terrain that maximizes a difficulty fitness for one combination of descriptor values. Tested on RL controllers for Cassie and ANYmal in simulation. The archive exposes distinct failure modes; one summary cites a consistent heading error on a narrow platform and obstacles that cause leg collisions or trap the robot. The terrains can also be used to improve the controllers. One summary reports an archive of 930 terrains — [arXiv abs](https://arxiv.org/abs/2506.01362); [themoonlight review](https://www.themoonlight.io/review/generating-diverse-challenging-terrains-for-legged-robots-using-quality-diversity-algorithm)
- [FOUNDATIONAL, 2020] Fontaine & Nikolaidis: MAP-Elites beat Monte Carlo simulation and optimization-based methods at covering the failure-scenario space for shared-autonomy robots — [ar5iv 2012.04283](https://ar5iv.labs.arxiv.org/html/2012.04283)
- Caveats for QD: standard QD methods struggle with noisy evaluations (Flageat & Cully), and the choice of behavior descriptor can leave out the dynamics that matter — [search result set incl. arXiv 2305.01264](https://arxiv.org/pdf/2305.01264)

**Scenario-testing comparability**
- "On the Adversarial Scenario-based Safety Testing of Robots: the Comparability and Optimal Aggressiveness" (arXiv 2209.09879). Studies how testing algorithms compare when they probe a Cassie biped with push-over disturbances, using both an RL controller and a hybrid-zero-dynamics controller — [arXiv PDF](https://arxiv.org/pdf/2209.09879)

### Inferences
- **Sample efficiency ranking (my synthesis; no paper benchmarks all of these on locomotion).** For low-dimensional parametric disturbances (push magnitude/direction/timing/gait phase, friction, payload, motor strength, latency, noise scale; about 3-12D):
  - BO or TuRBO is the most sample-efficient per rollout but scales poorly.
  - CEM and CMA-ES are robust, trivially parallel, and need hundreds to a few thousand rollouts.
  - Random search is the baseline every paper should report.
  - For time-varying attacks (sequences), learned RL adversaries or AST are needed, but they cost about one training run.
  - For *coverage* of distinct failure modes, MAP-Elites beats optimizers (Fontaine & Nikolaidis; QD-terrain 2025).
- **CPU-only recipe (implementable in a few hours with MuJoCo + PyTorch).**
  - Parameterize a scenario θ = [push_impulse_Ns, push_dir_rad, push_phase, friction μ, payload_kg, payload_offset, Kp_scale/motor_strength per leg, actuator_fault_leg_id+gain, action_latency_steps, obs_noise_std].
  - Define a continuous robustness score so the optimizer gets a gradient-like signal, not just pass/fail. Examples: min base height margin, max |roll|/|pitch| margin, or time-to-fall; this is the analog of STL "trace robustness" used in CEM falsification.
  - Search with CEM (Gaussian, 50-100 samples per iteration, top 10-20% elites, 10-20 iterations), or with the `cma` pip package.
  - Add a cost penalty on ‖θ‖ (the analog of the H-infinity / "critical attack" cost constraint), so the search returns *minimal* failures and not trivially huge pushes.
- A MAP-Elites grid over 2 descriptors (e.g., push direction × friction, or terrain roughness × step height) with the fitness "1 − normalized survival" is roughly 100 lines of numpy and yields a visual failure atlas.
- Low-magnitude sequential observation/command attacks (Shi et al.) are a failure axis that pure physical-parameter sweeps miss, so a test suite should include at least one bounded observation-noise adversary.

### Gaps
- No head-to-head numbers were found comparing CEM vs CMA-ES vs BO vs RL adversaries on the same locomotion policy. Exact attack success rates and fine-tuning gains for Shi et al. 2024 were not retrievable (arXiv was blocked).
- Which black-box method won in the 2022 joint-attack paper (random vs differential evolution) was not confirmed.
- The specific findings of 2209.09879 (optimal aggressiveness) were not retrieved.

## Q2. How do papers define and measure robustness margins?

### Takeaway
The most common scalar margin is **maximum recoverable push impulse (or force)**. It is found by a per-direction line search or a fixed-step ramp, with push duration, location and gait phase fixed and reported. Other margins are curves of success rate (or achieved velocity) against a swept parameter, and survival thresholds such as the "tipping threshold". Gait phase strongly affects push recoverability, so it must be controlled. A legged-robot disturbance-rejection test method (ASTM WK86916) is in progress but not yet published.

### Cited Findings
- Cassie sim-to-real (arXiv 2204.04340, 2022) uses a **line search in each direction** to find the largest force the policy can resist. The impulse lasts 0.2 s and is applied at the pelvis, with 3.0 s allowed for recovery. Per-direction results are averaged into one number — [arXiv PDF](https://arxiv.org/pdf/2204.04340)
- A bipedal whole-body-control paper **ramps the impulse in fixed steps** until the robot can no longer recover. It notes that pushes at certain phases are hardest because it takes about the entire swing duration to make a new touchdown — [arXiv 2109.07196](https://arxiv.org/pdf/2109.07196)
- Quadruped stabilization "is dependent on the instantaneous gait phase", so push timing must be controlled or reported — [search result set, Go2/push query](https://arxiv.org/pdf/2503.10401)
- Humanoid hierarchical MPC (arXiv 2409.14342) reports a 131% average improvement in **maximum recoverable impulse** over baselines in simulation — [arXiv html](https://arxiv.org/html/2409.14342v3)
- FACET (arXiv 2505.06883, Go2 among its platforms) reports robustness to impulses up to 200 N·s and an 80% reduction in collision impulse in simulation — [arXiv html](https://arxiv.org/html/2505.06883v1)
- Robustness is also reported as **achieved velocity under an unmodeled actuator change**. In the joint-torque perturbation-injection humanoid paper (arXiv 2504.06585), standard DR reached 0.00 m/s and the proposed method 0.36 m/s under unmodeled actuator stiffness — [arXiv PDF](https://arxiv.org/pdf/2504.06585)
- An asymmetric adversarial "sabotage" method on Unitree Go2 claims a **50% higher physical tipping threshold** than standard DR. This comes from a sciforum conference listing with an unusual date and is unverified — [sciforum](https://sciforum.net/paper/35129)
- ASTM F45.06 (legged robots subcommittee, launched 2023, chaired by Bowen Weng) proposed work item **WK86916, "test method for disturbance rejection testing of legged robots"**. Stability and balance testing is its main focus. It is a proposed work item, not a published standard — [ASTM news](https://www.astm.org/news/stability-performance-legged-robots-so23); [The Robot Report](https://www.therobotreport.com/astm-announces-legged-robotics-subcommittee/); [Robotics247](https://www.robotics247.com/article/astm_international_proposes_test_method_legged_robots)
- Dawson & Fan and Bayesian Safety Validation estimate the **failure probability** or failure distribution under a prior over disturbances, which is a probabilistic margin rather than a worst-case one — [PMLR](https://proceedings.mlr.press/v229/dawson23a.html); [arXiv 2305.02449](https://pdf.arxiv.org/pdf/2305.02449)

### Inferences
- **Implementable margin protocol.** For each of N directions (e.g., 8) and M gait phases (e.g., 4), run a **bisection** on impulse magnitude (about 8 steps → about 0.5% resolution), with K seeds per probe (K≥3, success = majority). Report per-direction max recoverable impulse as a polar plot, plus its minimum (worst-case margin) and mean. That costs roughly 8×4×8×3 ≈ 770 short rollouts per policy, which is feasible on a CPU.
- **Robustness curves.** Sweep one factor at a time (friction μ, payload kg, motor strength %, latency ms, obs-noise σ, single-joint torque loss %). Plot success rate and velocity-tracking error. Define the "breaking point" as the value where success drops below 50% (or 90%), found by bisection on the success-rate curve.
- **Failure frontier.** On 2D slices (e.g., friction × push), use bisection along rays or a MAP-Elites grid to draw the boundary. Comparing frontiers before and after adversarial fine-tuning is the key visual.
- Bisection assumes monotonicity. Policies can fail non-monotonically (e.g., gait-phase effects), so pair bisection with random or CEM probes inside the "safe" region to catch islands of failure.

### Gaps
- No standard choice of push duration, application point or recovery window across quadruped papers was found. The ASTM WK86916 draft details are not public in the sources found.
- No confirmed real-world Go2 maximum-impulse figures were found.

## Q3. How are discovered failures fed back into training, and what gains are reported?

### Takeaway
There are three feedback patterns: (1) joint adversarial training with a cost-bounded learned disturber (H-infinity 2024, PL-CAP 2025, RARL); (2) post-hoc fine-tuning on discovered adversarial scenarios (Shi et al. 2024: robustness to the found attacks improves, but untargeted attack spaces stay vulnerable); and (3) curricula and environment design: regret-based UED (PAIRED, PLR, ACCEL), QD-generated terrain archives, and adaptive or bounded expansion of domain-randomization ranges (ADR, HORIZON 2026). Published gains are mostly qualitative or tied to one benchmark (e.g., ACCEL about 3x the best baseline on BipedalWalker).

### Cited Findings
- Fine-tuning on Shi et al.'s adversarial scenarios "notably" increased robustness, including against sensor noise and external disturbances. However, the controllers remained vulnerable to attack spaces that were not targeted — [arXiv abs](https://arxiv.org/abs/2405.12424); [themoonlight review](https://www.themoonlight.io/review/rethinking-robustness-assessment-adversarial-attacks-on-learning-based-quadrupedal-locomotion-controllers)
- H-infinity locomotion: methods without the H-infinity framework are reported as highly vulnerable to deliberate adversarial disturbances. Joint policy and disturber training transferred to real Unitree robots — [mlanthology](https://mlanthology.org/corl/2024/long2024corl-learning); [HF papers](https://huggingface.co/papers/2404.14405)
- PL-CAP (humanoid G1): bounded-cost attacker during training gave better difficult-terrain traversal and more agile tracking — [arXiv 2507.08303](https://arxiv.org/html/2507.08303v1)
- [FOUNDATIONAL, ICML 2022] ACCEL (regret-based UED that edits previously high-regret levels). On BipedalWalker rough terrain it reached close to 75% of optimal performance, almost 3x the best baseline (PLR⊥). It used 2.07B environment steps, under 0.5% of what POET used. PLR finds challenging levels only through random search, while ACCEL builds on found structures. Harder levels can push agents toward a local optimum of "not falling" instead of walking forward. PAIRED estimates regret as the return gap between the student and a second agent — [ACCEL ar5iv](https://ar5iv.labs.arxiv.org/html/2203.01302); [accelagent.github.io](https://accelagent.github.io); [Jiang thesis](https://arxiv.org/pdf/2312.03126); open-source DCD code incl. PAIRED/PLR/ACCEL BipedalWalker configs — [GitHub AOS55/dcd](https://github.com/AOS55/dcd)
- QD terrain archive (2025) was also used to improve RL controllers. Size of the improvement was not visible in snippets — [arXiv 2506.01362](https://arxiv.org/abs/2506.01362)
- HORIZON (arXiv 2606.05143, 2026 preprint, quadruped). Widens the randomization range only within the policy's current *recoverable boundary*, and rolls back to a checkpoint after an unrecoverable step. It reports that widening directly beyond the frontier can collapse training, and that wider dynamics randomization is not always beneficial — [arXiv html](https://arxiv.org/html/2606.05143)
- A TU Darmstadt expert-lab report extended performance-based curricula to scale DR ranges and reward penalties (building on OpenAI ADR). It reported strongly increased sample efficiency in a simulated box-climbing task — [TU Darmstadt PDF](https://www.ias.informatik.tu-darmstadt.de/uploads/Team/OlegArenz/Leon_Magnus_Expert_Lab.pdf)
- Risk-averse policy learning for quadrupeds (arXiv 2308.09405) compares PPO, wider-DR PPO and a risk-averse method for push robustness — [arXiv PDF](https://arxiv.org/pdf/2308.09405)
- DrEureka (arXiv 2406.01967) uses LLMs to generate DR parameter ranges for quadruped sim-to-real — [arXiv html](https://arxiv.org/html/2406.01967v1)
- Joint-torque-space perturbation injection (humanoid, arXiv 2504.06585) beat standard DR under unmodeled actuator stiffness (0.36 vs 0.00 m/s) — [arXiv PDF](https://arxiv.org/pdf/2504.06585)

### Inferences
- **Cheapest closed loop for a small project ("find → replay → fine-tune → re-measure").**
  1. Run CEM or MAP-Elites to collect failing θ's into a buffer.
  2. Fine-tune PPO with a sampling mix, e.g., 50% nominal DR, 30% replay of discovered failures (PLR-style, prioritized by failure score or TD-error), and 20% newly widened DR.
  3. Re-run the *same* search plus a *held-out* attack family to detect overfitting to the found attacks (Shi et al. show untargeted spaces stay vulnerable).
- Bound attack strength with a cost penalty, or keep θ inside the current recoverable boundary (HORIZON, H-infinity, PL-CAP all bound the adversary). Unbounded adversaries produce unlearnable scenarios and collapse training, or teach "don't move" behaviors (ACCEL's note).
- Always report nominal-task performance (velocity tracking) after hardening, to catch robustness-performance tradeoffs.

### Gaps
- No quantified before/after numbers were available for Shi et al. fine-tuning, the H-infinity disturber versus DR, or QD-terrain fine-tuning.
- No 2023-2026 paper applying PAIRED/PLR/ACCEL to 3D quadruped/humanoid locomotion with reported gains was found within the search budget. The 50% tipping-threshold claim (sciforum) is unverified.

## Q4. Any metrics or standards for "deployment readiness" of locomotion policies?

### Takeaway
No agreed deployment-readiness standard exists yet. ASTM F45.06 is drafting legged-robot stability/disturbance-rejection test methods (WK86916, proposed). In practice, labs gate deployment on a **sim-to-sim transfer check** (train in Isaac Gym/Lab → evaluate zero-shot in MuJoCo → hardware), on velocity-tracking RMSE, and on robustness sweeps. Known pitfalls are simulator-parameter artifacts (armature, friction).

### Cited Findings
- ASTM F45.06 subcommittee (2023) aims to define measurable locomotion stability testing. Its first proposed item is WK86916 (disturbance rejection). It is not yet a published standard — [ASTM](https://www.astm.org/news/stability-performance-legged-robots-so23); [Automation.com](https://www.automation.com/article/astm-international-subcommittee-legged-robotics); NIST mobility test-method work with ASTM F45/F48 — [NIST](https://www.nist.gov/node/1535946)
- Common pipeline: Isaac Lab training → zero-shot MuJoCo → Unitree G1 (RoboDreamer, arXiv 2609.07096). Humanoid-Gym uses Isaac Gym → MuJoCo → real, with MuJoCo as the pre-hardware validation step — [arXiv 2609.07096](https://arxiv.org/pdf/2609.07096)
- MuJoCo is described as closer to reality in contact and joint physics than Isaac Gym. A zero-armature setting used in Isaac Gym training caused failure in MuJoCo — [arXiv 2607.11624](https://arxiv.org/pdf/2607.11624)
- Velocity-tracking RMSE (linear and yaw) in MuJoCo is used as a pre-deployment metric. Tuned MuJoCo friction can make a policy look smoother without making it more robust — [arXiv 2512.18938](https://arxiv.org/pdf/2512.18938)
- The SIMPLE benchmark (arXiv 2606.08278) couples MuJoCo physics at 500 Hz with Isaac Sim rendering and reports that simulation rankings closely track real-world performance. It targets loco-manipulation — [arXiv 2606.08278](https://arxiv.org/pdf/2606.08278)
- Survey of black-box safety validation lists the three validation tasks: falsification, most-likely failure, and failure-probability estimation — [arXiv 2005.02979](https://arxiv.org/pdf/2005.02979); [Bayesian Safety Validation](https://pdf.arxiv.org/pdf/2305.02449)

### Inferences
- A defensible "readiness scorecard" for a small project (my synthesis, not a standard):
  1. Nominal: success rate and velocity-tracking RMSE across a command grid.
  2. Margins: max recoverable impulse (per direction, worst-case and mean) and 50%/90% breaking points for friction, payload, motor strength, latency and obs-noise.
  3. Adversarial: success rate under a fixed-budget CEM/MAP-Elites search (e.g., "no failure found in 1000 rollouts within ‖θ‖≤B") and under a bounded observation-noise adversary.
  4. Probabilistic: estimated failure probability under the deployment disturbance prior, with a confidence interval (e.g., a Clopper-Pearson bound from N Monte Carlo rollouts; zero failures in N trials gives about a 3/N upper bound at 95%).
  5. Cross-simulator: the same metrics under a perturbed simulator configuration (armature, contact params, timestep) as a proxy for sim-to-sim transfer.
- Report the search budget alongside "no failure found", because falsification can only prove the existence of failures, never their absence.

### Gaps
- No content of ISO 13482 relevant to legged RL policies was found. No published numeric pass/fail thresholds (e.g., minimum impulse for certification) were found.
- No industry (Unitree/ANYbotics/Boston Dynamics) public readiness criteria were found within the search budget.
