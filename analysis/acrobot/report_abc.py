"""Reproduce descriptive Acrobot ABC tables/figures from frozen summaries.

Optional --raw-dir verifies C's original train/ and eval/ CSVs, not committed.
Only stdlib CSV parsing, NumPy and Matplotlib are needed (no pandas DLL).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from contextlib import suppress
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            with suppress(ValueError):
                row[key] = float(value) if value else None
    return rows


def write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select(rows, **filters):
    return [r for r in rows if all(r[k] == v for k, v in filters.items())]


def values(rows, key="mean_return"):
    return np.array([r[key] for r in rows], dtype=float)


def value(rows, seed, condition):
    group = select(
        rows,
        model="vqc",
        seed=seed,
        eval_noise_type=condition,
        eval_noise_level=0 if condition == "none" else 0.01,
    )
    assert len(group) == 1
    return group[0]["mean_return"]


def save_figure(fig, output, name):
    for ext in ("png", "svg"):
        path = output / f"{name}.{ext}"
        fig.savefig(path, dpi=300, metadata={"Date": None} if ext == "svg" else None)
        if ext == "svg":
            path.write_text(
                "\n".join(
                    line.rstrip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    base = root / "results/acrobot"
    cdir = base / "experiment_C"
    output = base / "overview"
    output.mkdir(parents=True, exist_ok=True)
    a = read(base / "experiment_A/evaluation_per_seed.csv")
    b = read(base / "experiment_B/evaluation_per_seed.csv")
    c = read(cdir / "evaluation_per_seed.csv")
    train = read(cdir / "training_per_seed.csv")
    agg = read(cdir / "evaluation_across_seeds.csv")
    assert [len(a), len(b), len(c), len(train)] == [15, 85, 6, 3]
    assert all(
        r["episodes"] == 100
        and r["experiment"] == "C"
        and r["train_noise"] == "depolarizing=0.01"
        for r in c
    )
    assert all(r["episodes"] == 600 for r in train)
    assert {r["seed"] for r in c} == {0, 1, 2}
    assert {r["eval_noise_type"] for r in c} == {"none", "depolarizing"}
    assert (
        len({(r["seed"], r["eval_noise_type"], r["eval_noise_level"]) for r in c}) == 6
    )
    for row in agg:
        group = values(select(c, eval_noise_type=row["eval_noise_type"]))
        assert len(group) == row["count"] == 3
        np.testing.assert_allclose(
            [group.mean(), group.std(ddof=1)], [row["mean"], row["std"]]
        )

    if args.raw_dir:
        for seed in range(3):
            name = f"Acrobot-v1_vqc_{seed}_depolarizing0.01.csv"
            raw = read(args.raw_dir / "eval" / name)
            assert len(raw) == 200
            assert (
                len(
                    {
                        (r["eval_noise_type"], r["eval_noise_level"], r["episode"])
                        for r in raw
                    }
                )
                == 200
            )
            for condition in ("none", "depolarizing"):
                group = select(raw, eval_noise_type=condition)
                assert len(group) == 100 and {r["episode"] for r in group} == set(
                    range(100)
                )
                np.testing.assert_allclose(
                    values(group, "return").mean(), value(c, seed, condition)
                )
            raw = read(args.raw_dir / "train" / name)
            assert len(raw) == 600 and [r["episode"] for r in raw] == list(range(600))
            row = select(train, seed=seed)[0]
            rolling = np.convolve(
                values(raw, "return"), np.ones(100) / 100, mode="valid"
            )
            np.testing.assert_allclose(
                [values(raw[-100:], "return").mean(), rolling.max(), raw[-1]["wall_s"]],
                [row["final100"], row["best100"], row["wall_s"]],
            )
            assert not (rolling >= -100).any() and row["solved_episode"] is None

    paired = []
    for seed in range(3):
        clean, noisy = value(b, seed, "none"), value(b, seed, "depolarizing")
        cclean, cnoisy = value(c, seed, "none"), value(c, seed, "depolarizing")
        paired.append(
            dict(
                seed=seed,
                B_clean=clean,
                B_depolarizing_p001=noisy,
                C_clean=cclean,
                C_depolarizing_p001=cnoisy,
                delta_clean=cclean - clean,
                delta_depolarizing_p001=cnoisy - noisy,
            )
        )
    write(cdir / "paired_BC_common_seeds.csv", paired)
    summary = []
    for model in ("mlp_large", "mlp_small", "vqc"):
        group = values(select(a, model=model))
        summary.append(
            dict(
                stage="A",
                model=model,
                train_noise="none",
                eval_noise="none",
                n=len(group),
                mean=group.mean(),
                sd=group.std(ddof=1),
            )
        )
    for source, stage, noise, condition in [
        (b, "B", "none", "depolarizing"),
        (c, "C", "depolarizing=0.01", "depolarizing"),
        (c, "C", "depolarizing=0.01", "none"),
    ]:
        group = values(
            select(
                source,
                eval_noise_type=condition,
                eval_noise_level=0 if condition == "none" else 0.01,
            )
        )
        assert len(group) in (3, 5)
        summary.append(
            dict(
                stage=stage,
                model="vqc",
                train_noise=noise,
                eval_noise="none" if condition == "none" else "depolarizing=0.01",
                n=len(group),
                mean=group.mean(),
                sd=group.std(ddof=1),
            )
        )
    write(output / "ABC_summary.csv", summary)

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "SC4055-Acrobot-ABC",
        }
    ):
        fig, axes = plt.subplots(
            1, 3, figsize=(12, 4.7), layout="constrained", sharey=True
        )
        ax = axes[0]
        for x, model in enumerate(["mlp_large", "mlp_small", "vqc"]):
            group = values(sorted(select(a, model=model), key=lambda r: r["seed"]))
            ax.scatter(
                x + np.linspace(-0.09, 0.09, len(group)),
                group,
                c="#0072B2",
                s=22,
                alpha=0.7,
            )
            ax.errorbar(
                x, group.mean(), yerr=group.std(ddof=1), fmt="D", c="black", capsize=4
            )
        ax.set_xticks(
            range(3),
            ["Large MLP\n4803 params", "Small MLP\n103 params", "VQC\n99 params"],
        )
        ax.set_title("A. Clean evaluation (5 seeds)")
        ax.set_ylabel("Mean raw return per seed (higher is better)")
        for ax, condition, title in [
            (axes[1], "depolarizing_p001", "C vs B. Noise p = 0.01"),
            (axes[2], "clean", "C vs B. Clean evaluation"),
        ]:
            left, right = (
                values(paired, "B_" + condition),
                values(paired, "C_" + condition),
            )
            jitter = np.linspace(-0.035, 0.035, 3)
            for i in range(3):
                ax.plot(
                    [jitter[i], 1 + jitter[i]],
                    [left[i], right[i]],
                    c="#999999",
                    alpha=0.65,
                    lw=1,
                )
            for x, group, color, marker in [
                (0, left, "#0072B2", "o"),
                (1, right, "#D55E00", "^"),
            ]:
                ax.scatter(x + jitter, group, c=color, marker=marker, s=35, zorder=3)
                ax.errorbar(
                    x + 0.08,
                    group.mean(),
                    yerr=group.std(ddof=1),
                    fmt="D",
                    c="black",
                    capsize=4,
                )
            ax.set_xticks([0, 1], ["Clean-trained\nB", "Noise-trained\nC"])
            ax.set_xlim(-0.3, 1.35)
            ax.set_title(title + "\nCommon seeds 0-2")
        for ax in axes:
            ax.set_ylim(-550, 0)
            ax.axhline(-500, c="#777777", ls=":", lw=1)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Acrobot: clean baselines and noise-aware training", fontsize=14)
        fig.supxlabel(
            "Dots: seed means over 100 episodes. Black diamonds: mean ± sample SD. Lines join matching seeds.\n"
            "Dotted line: -500 timeout floor. Horizontal dot offsets separate seeds; B/C use common seeds 0-2.",
            fontsize=9,
        )
        save_figure(fig, output, "ABC_overview")
        plt.close(fig)

        conditions = [
            ("none", 0),
            ("shots", 1000),
            ("shots", 100),
            ("shots", 10),
            ("coherent", 0.05),
            ("coherent", 0.1),
            ("coherent", 0.2),
            ("depolarizing", 0.001),
            ("depolarizing", 0.01),
            ("depolarizing", 0.05),
            ("depolarizing", 0.1),
            ("readout", 0.01),
            ("readout", 0.02),
            ("readout", 0.05),
            ("preset", "heron2026"),
            ("preset", "nisq2022"),
            ("preset", "stress"),
        ]
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        labels = []
        for y, (kind, level) in enumerate(conditions):
            group = values(
                sorted(
                    select(b, eval_noise_type=kind, eval_noise_level=level),
                    key=lambda r: r["seed"],
                )
            )
            assert len(group) == 5, (kind, level)
            ax.scatter(
                group, y + np.linspace(-0.1, 0.1, 5), s=20, color="#0072B2", alpha=0.65
            )
            ax.errorbar(
                group.mean(), y, xerr=group.std(ddof=1), fmt="D", c="black", capsize=3
            )
            labels.append("Clean reference" if kind == "none" else f"{kind} = {level}")
        ax.set_yticks(range(17), labels)
        ax.invert_yaxis()
        ax.set_xlim(-550, 0)
        ax.axvline(-500, c="#777777", ls=":")
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Mean raw return per seed (higher is better)")
        ax.set_title(
            "B. Deployment noise sensitivity: 5 clean-trained VQCs\nDots: 100-episode seed means; diamonds: mean ± sample SD",
            fontsize=11,
        )
        save_figure(fig, output, "B_noise_sweep")
        plt.close(fig)

    provenance = dict(
        replication="Training seed; 100 evaluation episodes per seed/condition",
        uncertainty="Sample standard deviation, ddof=1; no significance test",
        common_seeds=[0, 1, 2],
        paired_mean_delta={
            k: values(paired, k).mean()
            for k in ("delta_clean", "delta_depolarizing_p001")
        },
        raw_C_verified=args.raw_dir is not None,
        input_hash_normalization="CRLF converted to LF before SHA-256",
        training_minutes=[r["wall_s"] / 60 for r in train],
        files={
            p.relative_to(root).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")
            ).hexdigest()
            for folder in ("experiment_A", "experiment_B", "experiment_C")
            for p in sorted((base / folder).glob("*.csv"))
        },
        figure_dimensions_inches={"ABC_overview": [12, 4.7], "B_noise_sweep": [8, 7]},
        raster_dpi=300,
        matplotlib=matplotlib.__version__,
        numpy=np.__version__,
    )
    (output / "figure_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(dict(paired=paired, validation=provenance), indent=2))


if __name__ == "__main__":
    main()
