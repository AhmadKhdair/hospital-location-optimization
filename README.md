# Hospital Location Optimization — Hill Climbing & Simulated Annealing

Solves a facility-location problem (choosing hospital sites to minimize weighted travel cost for a population, penalized by a per-hospital cost `lambda`) using two local-search algorithms: best-improvement Hill Climbing and Simulated Annealing. Built for the AI course (ENCS3340) at Birzeit University.

Authors: Ahmad Khdair (1230500), Malek Zeghari (1230358)

## What it does

- Generates (or loads) a population of weighted demand points and a set of candidate hospital sites.
- For each `lambda` in `[1, 10, 50, 100]`, runs both Hill Climbing and Simulated Annealing multiple times (10 runs each, fixed seed for reproducibility) to select the subset of candidate sites that minimizes `total travel cost + lambda * number of hospitals opened`.
- Aggregates results (mean cost, best cost, variance, runtime) and produces comparison plots across lambda values.
- Runs a parameter-tuning sweep (Hill Climbing's initial selection rate, Simulated Annealing's cooling rate) for a fair side-by-side comparison.

## Usage

Random mode (generates a reproducible synthetic problem):
```bash
python project.py
```

File mode (load a specific population/candidates dataset):
```bash
python project.py --population population.csv --candidates candidates.csv
```

`population.csv` columns: `x, y, weight`. `candidates.csv` columns: `x, y`.

## Requirements

```
numpy
matplotlib
```

## Results

See `Report_1230500_1230358.pdf` for the full write-up and analysis. Sample output plots and data from a full run are in `results/`:

- `hc_lambda_*.png` / `sa_lambda_*.png` — best solution found by each algorithm at each lambda value
- `cost_vs_lambda.png`, `hospitals_vs_lambda.png`, `travel_vs_lambda.png`, `variance_vs_lambda.png` — comparison metrics across lambda
- `all_runs.csv` — raw results from every run
- `summary.csv` — aggregated statistics (mean/best/variance/runtime) per algorithm per lambda
- `parameter_tuning.csv` — results of the parameter-tuning sweep
