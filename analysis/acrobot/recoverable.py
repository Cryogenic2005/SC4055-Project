"""Persistent A/B/C execution; completed training seeds and eval episodes skipped.

Always use a mounted Google Drive path. A partially completed training seed
restarts from its beginning (optimizer/replay state is not checkpointed).
Evaluation resumes at the next completed-episode boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("A", "B", "C", "ABC"), default="ABC")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if not str(out).startswith("/content/drive/MyDrive/"):
        ap.error("Persistent output must be under mounted /content/drive/MyDrive/")
    if not os.path.ismount("/content/drive"):
        ap.error("Drive is not mounted; refusing to start computation")
    out.mkdir(parents=True, exist_ok=True)
    test = out / "WRITE_READ_TEST.json"
    test.write_text('{"persistent": true}', encoding="utf-8")
    if json.loads(test.read_text())["persistent"] is not True:
        raise OSError("Persistent storage verification failed")
    repo = Path(__file__).resolve().parents[2]
    sources = {
        str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((repo / "src").rglob("*.py"))
    }
    manifest = dict(
        schema=2,
        source_sha256=sources,
        episodes=600,
        eval_episodes=100,
        sampling="joint-multinomial-from-jitted-probabilities-v1",
    )
    lock = out / "PROTOCOL.json"
    if lock.exists() and json.loads(lock.read_text()) != manifest:
        raise ValueError(
            "Code/protocol changed. Use a new output directory; do not mix runs."
        )
    lock.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(*items):
        subprocess.run([sys.executable, "-u", *map(str, items)], cwd=repo, check=True)

    stages = list("ABC") if args.stage == "ABC" else [args.stage]
    if not (out / "random.csv").exists():
        run(
            "-m",
            "sc4055_project.baseline",
            "Acrobot-v1",
            "--episodes",
            100,
            "--out",
            out / "random.csv",
        )
    for stage in stages:
        seeds = range(3 if stage == "C" else 5)
        models = ("vqc", "mlp_small", "mlp_large") if stage == "A" else ("vqc",)
        for seed in seeds:
            for model in models:
                name = f"Acrobot-v1_{model}_{seed}"
                if stage == "C":
                    name += "_depolarizing0.01"
                ckpt = out / "checkpoints" / name
                done = ckpt / "TRAIN_COMPLETE.json"
                print(f"\n[{stage}] {name}", flush=True)
                if stage in ("A", "C") and not done.exists():
                    config = [
                        "--model",
                        model,
                        "--seed",
                        seed,
                        "--episodes",
                        600,
                        "--run-dir",
                        out,
                    ]
                    if stage == "C":
                        config += ["--noise", "depolarizing=0.01"]
                    print(
                        "Training fresh seed; files are written directly to Drive",
                        flush=True,
                    )
                    run("-m", "sc4055_project.acrobot", "train", *config)
                    summary = json.loads((ckpt / "training_summary.json").read_text())
                    if summary.get("quick") or not (ckpt / "best.pkl").exists():
                        raise RuntimeError(
                            "Training did not produce a valid final result"
                        )
                    done.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                elif done.exists():
                    print(
                        "Completed training preserved; skipping retraining", flush=True
                    )
                else:
                    raise FileNotFoundError(f"Complete A before B: {ckpt}")
                conditions = ["none"] if stage == "A" else None
                if stage == "C":
                    conditions = ["none", "depolarizing=0.01"]
                command = [
                    "-m",
                    "sc4055_project.acrobot",
                    "evaluate",
                    "--checkpoint",
                    ckpt / "best.pkl",
                    "--episodes",
                    100,
                    "--run-dir",
                    out,
                ]
                if conditions:
                    command += ["--conditions", *conditions]
                run(*command)
                run("analysis/acrobot/summarize.py", out)
        (out / f"STAGE_{stage}_COMPLETE.json").write_text(
            json.dumps(dict(stage=stage, complete=True)), encoding="utf-8"
        )
        print(f"Stage {stage} complete and persisted: {out}", flush=True)


if __name__ == "__main__":
    main()
