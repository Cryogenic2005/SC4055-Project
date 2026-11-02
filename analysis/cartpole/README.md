# CartPole post-training analysis (P1)

This mirrors the other task analyses:
- compact training summary CSV
- checkpoint health diagnostics (`best.pkl` vs `final.pkl`)
- optional noise-aware sweep using the shared harness

## 1) Summarise completed runs

```bash
python analysis/cartpole/summarize_training.py \
  --glob "runs/train/CartPole-v1_*.csv" \
  --out analysis_out/cartpole/training_summary.csv
```

Key fields:
- `best100`: best rolling 100-episode mean
- `final100`: rolling 100-episode mean at the end
- `solved_ep`: first episode where rolling mean reaches 475 (`CartPole-v1` threshold)

## 2) Inspect value scale and policy collapse

```bash
python analysis/cartpole/inspect_weights.py \
  --checkpoints runs/checkpoints/CartPole-v1_vqc_0

python analysis/cartpole/inspect_weights.py \
  --checkpoints runs/checkpoints/CartPole-v1_vqc_0_depolarizing0.01
```

Compare `best.pkl` vs `final.pkl`:
- if `Q std` and `|w_out|` inflate while `final100` drops, that's late-run instability
- keep reporting `best.pkl` for evaluation

## 3) Noise-aware training sweep (Experiment C style)

Use the shared sweep script already used by other members.

```bash
python analysis/noise_aware_sweep.py \
  --env CartPole-v1 \
  --base depolarizing=0.01 \
  --scales 0 0.25 0.5 1 2 \
  --seeds 0 1 2 \
  --episodes 1500 \
  --eval-episodes 50 \
  --out analysis_out/cartpole_sweep

python analysis/plot_noise_aware.py \
  --summary analysis_out/cartpole_sweep/summary.csv \
  --env CartPole-v1 \
  --base-level 0.01
```

`scale=0` is the clean-train arm (Experiment B baseline).

## 4) Full congruent run set with other tasks

For CartPole deliverables, run at least:
- Experiment A: `vqc`, `mlp_small`, `mlp_large` × seeds `0..4`
- Experiment B: evaluate each best checkpoint on full condition menu
- Experiment C: noise-aware train at `depolarizing=0.01`, `coherent=0.1`, `preset=heron2026` × seeds `0..4`

Keep all outputs in task-specific subfolders under `analysis_out/`.
