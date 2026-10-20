# Particle Swarm Optimization simulation

This folder contains a from-scratch, NumPy-based implementation of **Particle
Swarm Optimization (PSO)** and an expert-level study guide.

## Run it

```bash
python -m pip install -r requirements.txt
python pso_simulation.py --function rastrigin --iterations 200
```

If plotting dependencies are not available, the optimization and CSV/JSON
artifacts can still be generated:

```bash
python pso_simulation.py --function sphere --dimensions 5 --no-plot
```

The default output folder contains:

- `<function>_history.csv` — one row per iteration.
- `<function>_result.json` — configuration, final answer, and histories.
- `<function>_pso.png` — convergence plus particle movement for 2-D runs.
- `<function>_pso.gif` — optional animation of particle movement, velocity
  arrows, trails, global best, and convergence.

## Create an animated visualization

For the most visual demonstration, run a two-dimensional objective with
`--animate`:

```bash
python pso_simulation.py \
    --function rastrigin \
    --iterations 120 \
    --animate \
    --output-dir outputs
```

This creates `outputs/rastrigin_pso.gif`. The left side shows the particles
moving over the objective landscape; white trails show their paths, yellow
arrows show velocity, and the cyan star is the current global best. The right
side shows the global-best objective decreasing over time. For very long runs,
control the file size with `--animation-frames` and playback speed with
`--animation-fps`.

Animation requires Matplotlib and Pillow, both included in `requirements.txt`.
The animation is intentionally limited to two dimensions because particle
movement cannot be displayed directly in a static 2-D plane for higher-
dimensional search spaces.

Read **[PSO_GUIDE.md](PSO_GUIDE.md)** for the derivation, implementation
walk-through, parameter tuning, use cases, limitations, worked examples, and
interview/viva preparation questions.
