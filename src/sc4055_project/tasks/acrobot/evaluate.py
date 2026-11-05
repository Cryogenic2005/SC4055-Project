from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import jax
import numpy as np

from sc4055_project.qrl.envs import make_env
from sc4055_project.qrl.logging_utils import EvalLogger
from sc4055_project.qrl.noise import NOISE_MENU, parse_noise

from .agent import TASK_ADAPTER
from .inference import probability_function, sampled_q, z_eigenvalues


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def evaluate(
    cfg, params, conditions=NOISE_MENU, n_episodes=100, run_dir="runs", eval_seed=20000
):
    """Resume by completed episode, with persistent per-condition records.

    CSV is a derived export. JSON condition records are authoritative and are
    atomically replaced after EVERY episode. Existing checkpoint/config/seed
    mismatches are errors. Finite-shot inference samples the actual joint
    quantum distribution, compiled once; it does not use analytic expectations.
    """
    if cfg.env_id != "Acrobot-v1" or n_episodes < 1:
        raise ValueError("Acrobot-v1 and positive evaluation episodes required")
    conditions = list(dict.fromkeys(["none", *conditions]))
    specs = [parse_noise(c) for c in conditions]
    root = Path(run_dir)
    path = root / "eval" / f"{cfg.run_name}.csv"
    state = root / "eval_progress" / cfg.run_name
    state.mkdir(parents=True, exist_ok=True)
    checksum = hashlib.sha256()
    for array in jax.tree_util.tree_leaves(params):
        array = np.asarray(array)
        checksum.update(str((array.shape, str(array.dtype))).encode())
        checksum.update(array.tobytes())
    protocol = json.loads(
        json.dumps(
            dict(
                schema=2,
                eval_seed=eval_seed,
                episodes=n_episodes,
                cfg=cfg.to_dict(),
                params_sha256=checksum.hexdigest(),
                shots_engine="joint-multinomial-from-jitted-probabilities-v1",
            )
        )
    )
    meta = state / "protocol.json"
    if meta.exists() and json.loads(meta.read_text()) != protocol:
        raise ValueError("Evaluation protocol differs; use a new output directory")
    atomic_json(meta, protocol)
    packets = {}
    for file in state.glob("condition_*.json"):
        packet = json.loads(file.read_text())
        if packet["condition"] in packets:
            raise ValueError("Duplicate condition records")
        values = packet["returns"]
        if len(values) > n_episodes or not np.isfinite(values).all():
            raise ValueError("Invalid saved evaluation episodes")
        packets[packet["condition"]] = packet

    def export_csv():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        with EvalLogger(
            temporary,
            task=cfg.env_id,
            model=cfg.model,
            experiment="B" if cfg.noise == "none" else "C",
            train_noise=cfg.noise,
            seed=cfg.seed,
        ) as logger:
            for condition, packet in sorted(packets.items()):
                spec = parse_noise(condition)
                for ep, ret in enumerate(packet["returns"]):
                    logger.episode(spec.eval_type, spec.eval_level, ep, ret)
        temporary.replace(path)

    results = {}
    for cond, spec in zip(conditions, specs, strict=True):
        packet = packets.setdefault(cond, dict(condition=cond, returns=[]))
        returns = packet["returns"]
        record = state / (
            "condition_" + hashlib.sha256(cond.encode()).hexdigest()[:16] + ".json"
        )
        if len(returns) == n_episodes:
            results[cond] = float(np.mean(returns))
            print(
                f"[saved, skipped] {cond}: {n_episodes}/{n_episodes}, mean {results[cond]:.2f}",
                flush=True,
            )
            continue
        start = time.monotonic()
        print(
            f"[{cfg.run_name}] {cond}: resume {len(returns)}/{n_episodes}", flush=True
        )
        shots_mode = bool(spec.shots) and cfg.model == "vqc"
        q_fn = (
            None
            if shots_mode
            else TASK_ADAPTER.make_q_fn(cfg, spec, seed=eval_seed + cfg.seed)
        )
        probs = probability_function(cfg, params) if shots_mode else None
        signs = z_eigenvalues(cfg.n_qubits, cfg.n_actions) if shots_mode else None
        env = make_env(cfg.env_id, max_steps=cfg.max_steps, **cfg.env_kwargs)
        try:
            for ep in range(len(returns), n_episodes):
                ep_seed = eval_seed + cfg.seed * 100_000 + ep
                obs, _ = env.reset(seed=ep_seed)
                key = jax.random.PRNGKey(ep_seed)
                rng = np.random.default_rng(ep_seed)
                total, done, steps = 0.0, False, 0
                while not done:
                    key, k = jax.random.split(key)
                    x = TASK_ADAPTER.encode_obs(obs)
                    if shots_mode:
                        q = sampled_q(probs(x), params["w_out"], signs, spec.shots, rng)
                    else:
                        q = np.asarray(q_fn(params, x[None], k)[0])
                    if not np.isfinite(q).all():
                        raise FloatingPointError("Non-finite Q values")
                    obs, reward, term, trunc, _ = env.step(int(np.argmax(q)))
                    total += float(reward)
                    done = term or trunc
                    steps += 1
                    if ep == len(returns) and steps == 100 and ep % 5 == 0:
                        print(f"  {cond}: episode {ep + 1}, 100 steps...", flush=True)
                returns.append(total)
                atomic_json(record, packet)
                if (ep + 1) % 5 == 0 or ep + 1 == n_episodes:
                    export_csv()
                    print(
                        f"  {cond}: {ep + 1}/{n_episodes}, return {total:.1f}, "
                        f"mean {np.mean(returns):.2f}, elapsed {(time.monotonic() - start) / 60:.1f} min; saved",
                        flush=True,
                    )
        finally:
            env.close()
            export_csv()
        results[cond] = float(np.mean(returns))
    export_csv()
    atomic_json(
        path.with_suffix(".json"), dict(protocol=protocol, mean_returns=results)
    )
    return results
