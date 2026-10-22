# Turn four mini-projects into one experiment

Keep the plan's best feature, which is that each person owns one task from start to finish, and drop its weakest feature, which is four separate projects that cannot be compared. The team should build **one shared JAX + PennyLane codebase**, run **one experimental protocol**, and ask **one research question** across four tasks of increasing size: **FrozenLake (4 qubits), CartPole (4 qubits), Acrobot (6 qubits) and LunarLander (a hybrid model, trained noise-free and evaluated under noise)**. A good question is: *"How much hardware-like noise can a VQC agent tolerate, how does that tolerance shrink as the task and circuit grow, and does training under noise win any of it back?"* This matters for the grade. The rubric gives 50% to quantum method, correctness and experimental design, and those marks reward a controlled comparison more than four separate demos. The plan is feasible. A jitted 4-qubit CartPole agent trains in minutes per seed, and noisy density-matrix training is cheap at 4 qubits. It becomes impractical at 8 qubits, at roughly 46 hours per seed. So "LunarLander noise-free vs noisy training" in its current form should not be attempted. The prior work supports realistic expectations. Skolik et al. found that CartPole agents still learn under depolarizing noise up to p ≈ 0.01, fail at p = 0.1, and match noise-free performance under an IBM-calibration-style noise model ([arXiv 2212.09431](https://arxiv.org/abs/2212.09431)). That gives the team a published result to reproduce and extend. All work should be finished by **Sunday 8 November**, leaving **9–15 November** as a buffer.

## Splitting by task is right for ownership but wrong as four separate projects

The split by task has real strengths. Every member has a clear deliverable, the work runs in parallel, and nobody waits on anyone else for the core results. The 15-minute video also gets an obvious structure: one segment per presenter. Those strengths are worth keeping.

The weaknesses come from running four separate projects. If each person writes their own agent, noise model and plotting code, the team builds the same infrastructure four times. Each copy picks up its own bugs and hyperparameter quirks. The final report then compares results that differ in encoding, optimizer, seeds and noise definitions as well as in task, so it cannot draw any cross-task conclusion. Markers will see four thin results rather than one substantial one. The loads are also very uneven. FrozenLake with basis encoding is close to trivial: Chen et al. reached reward ≈0.9 by about episode 198 with only **28 parameters** ([arXiv 1907.00397](https://arxiv.org/pdf/1907.00397)). LunarLander, by contrast, has never been convincingly solved by a pure VQC. The claimed successes use PPO with classical pre-processing (8→2 dimensions feeding 4 qubits) or classical heads after measurement ([arXiv 2501.07085](https://arxiv.org/html/2501.07085v1); [ar5iv 2203.14348](https://ar5iv.labs.arxiv.org/html/2203.14348)).

LunarLander under noise is also too expensive to train. Density-matrix simulation needs memory that grows as 4^n. In a micro-benchmark run during this research (PennyLane 0.45.1 with JAX on a 2-vCPU machine), an 8-qubit noisy batch gradient took about **5 seconds**, against about **10 ms** noise-free. That puts noisy LunarLander training at roughly 46 hours per seed, while noisy *evaluation* of a trained policy takes about 1–1.5 hours per noise level. These are order-of-magnitude figures from a single run, not published benchmarks.

The fix is to keep ownership by task while sharing everything else. All four people use the same agent class, training loop, noise module, evaluation harness and plotting script. Each person also owns one shared component, so the infrastructure is built once. The four tasks then become four points along one axis of growing difficulty, with the qubit count running 4 → 4 → 6 → 8. That axis is what lets the report make a single argument.

## Four tasks, sized from a sure win to a stretch goal

The four tasks below were chosen so that every person is very likely to produce a working noise-free agent. The two smallest tasks also allow full noise-aware training. One algorithm, **DQN in the style of Skolik et al.**, serves all four tasks. Its recipe is well documented:

- data re-uploading with RX encoding,
- trainable input scaling,
- RX/RY/RZ variational layers with a ring of entanglers,
- Z-expectation readouts multiplied by trainable output weights.

Skolik's ablation showed this combination matters. Neither re-uploading alone nor trainable input scaling alone solved CartPole within 5,000 episodes, and fixed output ranges were fragile ([Quantum 6, 720](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/)). The TensorFlow Quantum (TFQ) tutorial's version of this setup (4 qubits, 5 layers, about 94 parameters) reached an average of 500 at about episode 600 and quotes roughly **40 minutes** for DQN and **20 minutes** for REINFORCE ([TFQ tutorial](https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning)).

| Task (Gymnasium 1.x) | Qubits / encoding | Expected noise-free training per seed* | Noise-aware training? | What the literature reports |
|---|---|---|---|---|
| **FrozenLake-v1** (16 states, 4 actions; threshold 0.70) | 4 qubits; basis encoding of the 4-bit state index (bit b → RX(πb)); 2–5 layers; about 28–60 params | Seconds to a few minutes; noisy (density matrix) about 2–3× that | **Yes, full sweep** | Chen et al.: ≈0.9 reward by about episode 200, and 0.94–0.95 on the IBM device ibmq-valencia ([arXiv 1907.00397](https://arxiv.org/abs/1907.00397)). Skolik: 5, 10 and 15 layers all solve it ([Quantum](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/)) |
| **CartPole-v1** (4-dim, 2 actions; threshold 475) | 4 qubits; arctan-scaled RX re-uploading with trainable λ; 5 layers; about 62–94 params | About 1–3 min with JAX jit; about 70 min on the slow non-jit path; noisy about 2–3 min | **Yes, full sweep** | Skolik: best single agent solved in 206 episodes with 62 params; an MLP needed about half the episodes with about twice the params ([Quantum](https://quantum-journal.org/papers/q-2022-05-24-720/pdf/)). Under noise, solved for p ≤ 0.01 and failed at p = 0.1 ([arXiv 2212.09431](https://arxiv.org/abs/2212.09431)) |
| **Acrobot-v1** (6-dim, 3 actions; threshold −100) | 6 qubits, one per feature, re-uploading; 2–5 layers; about 90–180 params | About 5–15 min jitted; noisy about 50 min per 100k steps | **Partial: one condition, 3 seeds** | Jerbi REINFORCE with 6 qubits, 90/180 params, results shown only in plots ([arXiv 2103.05577](https://arxiv.org/pdf/2103.05577)). PPO-Q: comparable to classical PPO after a slower start ([arXiv 2501.07085](https://arxiv.org/html/2501.07085v1)) |
| **LunarLander-v3** (8-dim, 4 actions; threshold 200) | **Hybrid:** linear or small-MLP encoder 8→4 feeding a 4-qubit VQC (fallback: 8 qubits, one per feature) | About 10–20 min of circuit time per seed for 8 qubits, plus loop overhead; convergence is the real risk | **No: evaluation under noise only** | SVQC (8 qubits plus a classical head, PPO): about 220 average, best of 10 runs ([arXiv 2203.14348](https://arxiv.org/pdf/2203.14348)). PPO-Q (8→2 encoder, 4 qubits, 72 params) "accomplishes the task" ([arXiv 2501.07085](https://arxiv.org/html/2501.07085v1)) |

*Estimates come from the local benchmark plus TFQ's stated runtimes. Real training-loop overhead can double them.

### Why the hybrid 4-qubit version of LunarLander is the right choice

The hybrid design keeps the noisy part of the circuit at 4 qubits. That makes LunarLander's noise sweeps as cheap as CartPole's, and even one noise-aware training run becomes possible. The 8-qubit version, by contrast, is limited to a few noisy evaluation points. The cost is honesty about what the quantum part contributes. In PPO-Q the VQC holds only 24 of 72 actor parameters ([arXiv 2501.07085](https://arxiv.org/html/2501.07085v1)). The 2026 single-qubit CartPole result shows the same pattern: a 1-qubit circuit sits inside 32-neuron classical layers, which do most of the work ([arXiv 2605.01716](https://arxiv.org/pdf/2605.01716)).

The team should state this tradeoff in the report and include a **classical-encoder-only ablation**: the same model with the VQC replaced by a linear layer. That ablation shows whether the quantum layer carries any weight. The team should also expect LunarLander to stay **below 200**. A clearly reported partial result with a learning curve meets the rubric's "critical analysis" criterion. A missing result does not.

### Tasks to avoid

MountainCar fails mainly because its sparse reward makes exploration hard, which is not a quantum problem ([Jerbi et al.](https://arxiv.org/pdf/2103.05577)). Pendulum needs continuous-action PPO, and even its best VQC was beaten by a classical network ([arXiv 2312.13798](https://arxiv.org/pdf/2312.13798)).

## One protocol, one noise menu, one normalized score

### The shared codebase

The shared codebase is a single `QAgent` built in a pure-functional JAX style:

- an encoder,
- the VQC,
- a readout,
- an optional classical head,
- a config dict that sets `env_id`, `n_qubits`, `n_layers`, `encoding`, `readout`, `noise` and `seed`.

A `make_qnode(cfg)` function selects the simulator device. It uses `default.qubit` (the noise-free statevector simulator) for noise-free, shot and coherent-error runs, and `default.mixed` (the density-matrix simulator) when channel noise is present. All noise insertion goes through one `apply_noise()` helper.

The fast path matters. In the local benchmark, `default.qubit` with `jax.jit` and `jax.vmap` ran a 4-qubit forward pass in **0.36 ms**, against **10.7 ms** for plain autograd. Parameter-shift gradients cost 2× the number of parameters per evaluation. One PennyLane user's run estimate dropped from "weeks" to "hours" after switching away from parameter-shift ([PennyLane forum](https://discuss.pennylane.ai/t/using-pennylane-for-q-reinforcement-learning/1947)). The JAX transformations demo shows the same pattern: a jitted second call takes about 0.03 ms against 3.9 ms without jit ([PennyLane JAX demo](https://pennylane.ai/demos/tutorial_jax_transformations)). Use `optax.multi_transform` to give separate learning rates to the input scaling, the circuit angles and the output weights; the TFQ DQN uses 0.001, 0.001 and 0.1 ([TFQ tutorial](https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning)).

For the classical baselines, swap `forward` for an MLP and keep the same training loop and seeds. Two MLP baselines are needed:

- a **parameter-matched MLP** with about the same parameter count as the VQC,
- a **large MLP** (2×64 ReLU, a few thousand parameters).

Papers that report only the small baseline make quantum agents look better than they are. Well-tuned networks with about 4,400–4,700 parameters beat VQC agents with about 176–178 parameters on Pendulum and LunarLander ([arXiv 2312.13798](https://arxiv.org/abs/2312.13798)). The benchmarking study of 2025 explicitly says competitive CartPole results are not evidence of quantum advantage ([arXiv 2501.15893](https://arxiv.org/pdf/2501.15893)).

### The noise menu

Use the same noise levels on every task. They follow Skolik's four error classes and are anchored to 2024–26 hardware data.

| Condition | Levels | How to implement | Realism anchor |
|---|---|---|---|
| Shot noise | ∞, 1000, 100 (and 10 as a stress test) | `shots=N` on the device; evaluation only, except on FrozenLake | Q-learning with 100 fixed shots stayed below 175 on CartPole ([arXiv 2212.09431](https://arxiv.org/abs/2212.09431)); a hardware CartPole scored 143 at 128 shots and 500 at 1,024 shots ([arXiv 2605.01716](https://arxiv.org/pdf/2605.01716)) |
| Coherent over-rotation | σ ∈ {0.05, 0.1, 0.2} rad of Gaussian noise added to all angles, resampled for every circuit call | Statevector, essentially free | Performance degrades above σ ≈ 0.1; policy gradient is more robust ([arXiv 2212.09431](https://arxiv.org/abs/2212.09431)) |
| Depolarizing | p ∈ {0.001, 0.01, 0.05, 0.1} after every gate | `qml.DepolarizingChannel` on `default.mixed` | Solved for p ≤ 0.01, failed at p = 0.1 on CartPole ([arXiv 2212.09431](https://arxiv.org/abs/2212.09431)) |
| Readout | p ∈ {0.01, 0.02, 0.05} | `default.mixed(readout_prob=…)` ([PennyLane docs](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.DefaultMixed.html)) | Heron readout error 1.3–2.1×10⁻² ([arXiv 2410.00916](https://arxiv.org/pdf/2410.00916)) |
| **"Heron-2026" preset** | 1Q 3×10⁻⁴, 2Q 3×10⁻³, readout 1.5×10⁻², T1 ≈ 150 µs, T2 ≈ 100 µs, 1000 shots | Noise model plus `ThermalRelaxationError` | Heron r2 CZ 2.85×10⁻³, SX 2.88×10⁻⁴ ([arXiv 2410.00916](https://arxiv.org/pdf/2410.00916)); Heron r3 2Q 1.52×10⁻³ ([PostQuantum](https://postquantum.com/industry-news/ibm-heron-r3-pittsburgh/)) |
| **"NISQ-2022" preset** | 1Q 1×10⁻³, 2Q 1×10⁻², amplitude damping 3×10⁻⁴, readout 1×10⁻² | Same | Skolik config (a), which matched noise-free performance ([arXiv 2212.09431](https://arxiv.org/pdf/2212.09431)) |
| Stress preset | 10× the Heron-2026 preset | Same | Above Skolik's failure region |

Two implementation rules protect the 25% "correctness" mark.

The first rule is to **state the depolarizing convention**. A device's reported gate error is not the same number as the depolarizing parameter p; the two differ by a factor that depends on the Hilbert-space dimension. Pick one conversion and write it down. Also pull live Heron medians from the IBM Quantum Platform, because calibration numbers drift ([IBM QPU docs](https://quantum.cloud.ibm.com/docs/en/guides/qpu-information)).

The second rule is to **never combine shots with backprop**, because real hardware cannot backpropagate. Use either exact expectations with noise channels, or shots with parameter-shift gradients.

The course lists Qiskit, so the team can also earn a cheap cross-check. Build an Aer `NoiseModel.from_backend(...)`, import it with `qml.from_qiskit_noise`, and confirm that one circuit's expectation values match the hand-written PennyLane model. That importer drops readout errors with a warning, so re-add them by hand ([PennyLane Qiskit-noise demo](https://pennylane.ai/qml/demos/tutorial_how_to_import_qiskit_noise_models)). Unit tests should also confirm the textbook contractions. A symmetric readout flip scales ⟨Z⟩ by (1−2p), and depolarizing noise shrinks Pauli expectations toward zero.

### The protocol every task follows

Every task runs the same three experiments.

**Experiment A, noise-free training.** Train the VQC, the parameter-matched MLP and the large MLP, with **5 seeds each** (10 seeds for FrozenLake and CartPole if time allows). Skolik averaged 5–10 agents ([arXiv 2212.09431](https://arxiv.org/pdf/2212.09431)).

**Experiment B, train noise-free then evaluate under noise.** Take every trained VQC and run 100 greedy episodes under each noise condition. This needs forward passes only, so it is cheap, and it is the standard approach in hardware-transfer work ([arXiv 2203.14348](https://arxiv.org/pdf/2203.14348)).

**Experiment C, noise-aware training.** Retrain from scratch under the depolarizing 0.01, σ = 0.1 and Heron-2026 conditions. Use 5 seeds on FrozenLake and CartPole and one condition with 3 seeds on Acrobot. Then compare against Experiment B at the same noise level.

### Normalized score and reporting

Report a **normalized score** for each run:

> normalized score = (R − R_random) / (R_threshold − R_random)

Measure R_random yourselves over 100 episodes of a random policy. This normalization is our recommendation rather than a published standard, and it is what lets one plot hold all four tasks.

The team's headline figure is then **normalized score versus noise strength**, with one line per task. A second figure shows the **"noise tolerance" (the largest p that keeps 90% of the noise-free score) against qubit count**.

Report "solved" as a mean over the last 100 episodes compared with `env.spec.reward_threshold`, and say which threshold convention you used. Gymnasium registers 475 for CartPole-v1, but its docs say 500. LunarLander-v2 is now v3. Expect results that do not move steadily in one direction. Gross found that low amplitude damping *improved* one policy ([arXiv 2512.01502](https://arxiv.org/abs/2512.01502)). Report results like that rather than "fixing" them.

## A 17-day schedule with go/no-go gates

This schedule fits the plan because noise-free CartPole takes minutes when jitted. The compute budget is not the constraint; debugging is. The schedule therefore front-loads infrastructure and makes the protocol fixed by **Wednesday 28 October**.

| Dates | Milestone | Work |
|---|---|---|
| **Thu 22 Oct** | Kick-off | Agree the research question, this protocol and the roles. Create the Git repo and a pinned `requirements.txt` |
| **Fri 23 Oct** | Every machine passes a smoke test | Install pinned versions on every laptop and on Colab. Run a jitted 4-qubit QNode and `LunarLander-v3` |
| **Sat 24 – Sun 25 Oct** | Skeleton works | P1: `QAgent` and DQN loop solve CartPole on 1 seed. P2: noise module and unit tests. P3: evaluation harness and CSV logging. P4: MLP baseline running on all four environments, which validates the training loop |
| **Mon 26 – Wed 28 Oct** | **M1: protocol frozen** | Every task trains a noise-free VQC on 1 seed. Hyperparameters chosen with a short sweep (QRL is "highly sensitive to hyperparameters": [arXiv 2507.07593](https://arxiv.org/html/2507.07593v1)). Qiskit cross-check of the noise model passes |
| **Thu 29 Oct – Sun 1 Nov** | **M2: Experiment A done; LunarLander go/no-go on Sun 1 Nov** | Run 5+ seeds of VQC and both MLPs on each task. If LunarLander is not learning, fall back to "noise-free learning curve plus noisy evaluation of best checkpoint" |
| **Mon 2 – Wed 4 Nov** | **M3: Experiments B and C done** | Noise-sweep evaluations on all tasks. Noise-aware training on FrozenLake and CartPole, and one Acrobot condition |
| **Thu 5 Nov** | Analysis meeting | Aggregate the CSVs, build the cross-task figures, agree the 3–4 headline claims |
| **Fri 6 – Sun 8 Nov** | **M4: complete draft** | Report draft, slides, recorded 15-minute video (about 3 minutes shared intro and method, about 2.5 minutes per task, about 2 minutes cross-task conclusion), cleaned code with a README |
| **Mon 9 – Sun 15 Nov** | Buffer | Re-run failed seeds, fix figures, re-record if needed. **Submit by Fri 13 Nov**; keep the weekend as slack |

### Roles

Each person owns one task end to end plus one shared component. The lighter tasks are paired with the heavier shared jobs.

| Person | Task owned | Shared component | Report / video responsibility |
|---|---|---|---|
| P1 | CartPole (reference task; reproduces Skolik's noise thresholds; optional REINFORCE comparison, since policy gradient was more robust to coherent noise) | Core `QAgent`, DQN loop, JAX/optax setup | Method section; video intro |
| P2 | FrozenLake (full noise-aware sweep, including shot-noise training with parameter-shift) | Noise module, presets, Qiskit cross-check, unit tests | Noise-model section; **report editor** |
| P3 | Acrobot (6 qubits; tests how noise cost and tolerance scale) | Evaluation harness, logging, aggregation and plotting scripts | Results figures; **video editor** |
| P4 | LunarLander hybrid (plus the classical-encoder-only ablation) | Classical MLP baselines and env wrappers (including box2d) | Related work, limitations and conclusion |

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Unpinned `pip install pennylane jax` installs JAX 0.12.0.1, which crashes PennyLane 0.45.1 with `jax.core has no attribute is_concrete` | Pin `pennylane==0.45.*` and `jax[cpu]==0.10.2`. Both crash and pin were verified locally on 22 Oct 2026 |
| Box2D install fails | Run `pip install swig` before `gymnasium[box2d]` ([Gymnasium Box2D docs](https://gymnasium.farama.org/environments/box2d)). Windows users should switch to Colab or WSL on day 1 |
| A member ends up on the non-jit autograd path (about 70 min per CartPole seed instead of 1–3 min) or uses parameter-shift for training | Shared code makes the fast path the default. Wall-clock time is logged per run |
| Noisy simulation at 6–8 qubits is too slow (6-qubit noisy batch gradient about 257 ms; 8-qubit about 5 s, measured locally) | Noise-aware training only at 4 qubits. Evaluation-only noise at 6–8 qubits. Re-benchmark loop versus vmap on your own machine |
| Colab sessions disconnect | Save checkpoints every N episodes. Spread seeds across four people's Colab sessions |
| LunarLander never learns | Hybrid encoder, go/no-go on 1 Nov, and present it as an honest negative result |
| Fake-backend import paths in `qiskit_ibm_runtime` changed | Use `GenericBackendV2` as in the PennyLane demo ([demo](https://pennylane.ai/qml/demos/tutorial_how_to_import_qiskit_noise_models)), or the hand-written Heron preset |

Avoid TFQ, CUDA-Q and Catalyst unless someone already knows them. TFQ pins TensorFlow 2.18.1 with legacy Keras ([TFQ tutorial](https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning)). CUDA-Q's noisy path is based on trajectories and shot sampling ([CUDA-Q noisy simulators](https://nvidia.github.io/cuda-quantum/0.13.0/using/backends/sims/noisy.html)), and at 4–8 qubits it gives no gain over jitted JAX.

## Conclusion

Several outcomes would answer the research question. The team might find that tolerance to noise drops sharply from FrozenLake to Acrobot. It might find that argmax action selection protects Q-learning from the shrinking of expectation values caused by depolarizing and readout noise, which is how both Chen and Skolik explained their agents' robustness. Or it might find that noise-aware training recovers part of the lost score at 4 qubits. Any of these is a real finding. Reproducing a published threshold (p ≈ 0.01 on CartPole), extending it across task size, and testing it against realistic 2026 Heron error rates gives the team a defensible quantum-methods story.

Under the shared protocol, the Heron-2026 preset is about 3–10× cleaner than the 2022-era model Skolik already found harmless. So the team's most likely and most interesting result is this: *for these small circuits, realistic gate noise matters less than shot budget and circuit depth, and a well-sized classical MLP still wins*. That makes a stronger critical analysis than claiming a quantum advantage.

## References

**Papers**

- Skolik, Jerbi, Dunjko, "Quantum agents in the Gym" (Quantum 6, 720, 2022). https://quantum-journal.org/papers/q-2022-05-24-720/pdf/
- Skolik, Mangini, Bäck, Macchiavello, Dunjko, "Robustness of quantum RL under hardware errors" (EPJ QT 2023). https://arxiv.org/abs/2212.09431
- Jerbi et al., "Parametrized quantum policies for RL" (NeurIPS 2021). https://arxiv.org/pdf/2103.05577
- Chen et al., "Variational Quantum Circuits for Deep RL" (IEEE Access 2020). https://arxiv.org/abs/1907.00397
- Franz et al., "Uncovering Instabilities in Variational-Quantum Deep Q-Networks". https://arxiv.org/abs/2202.05195
- SVQC with PPO, unentangled agents including LunarLander. https://arxiv.org/pdf/2203.14348
- PPO-Q, hybrid quantum PPO including LunarLander. https://arxiv.org/html/2501.07085v1
- VQC design for continuous-environment QRL. https://arxiv.org/abs/2312.13798
- Benchmarking QRL. https://arxiv.org/pdf/2501.15893
- Chen, QA3C (asynchronous quantum RL). https://arxiv.org/pdf/2301.05096
- Gross, formal verification of noisy QRL policies. https://arxiv.org/abs/2512.01502
- Ngo et al. 2026, CartPole on quantum hardware. https://arxiv.org/pdf/2605.01716
- AbuGhanem, IBM hardware benchmarks. https://arxiv.org/pdf/2410.00916

**Code**

- CleanQRL (PennyLane + PyTorch, single-file agents). https://github.com/fhg-iisb/cleanqrl
- CleanQRL docs. https://fhg-iisb.github.io/cleanqrl-docs/
- Skolik DQN reproduction in PennyLane. https://github.com/n-toscano/qrl-dqn-gym

**Tutorials and docs**

- TFQ quantum RL tutorial. https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning
- PennyLane noise models demo. https://www.pennylane.ai/qml/demos/tutorial_how_to_use_noise_models
- PennyLane Qiskit-noise import demo. https://pennylane.ai/qml/demos/tutorial_how_to_import_qiskit_noise_models
- PennyLane JAX demo. https://pennylane.ai/demos/tutorial_jax_transformations
- PennyLane `default.mixed` docs. https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.DefaultMixed.html
- Gymnasium Box2D docs. https://gymnasium.farama.org/environments/box2d
- IBM QPU information guide. https://quantum.cloud.ibm.com/docs/en/guides/qpu-information
- CUDA-Q noisy simulators. https://nvidia.github.io/cuda-quantum/0.13.0/using/backends/sims/noisy.html
