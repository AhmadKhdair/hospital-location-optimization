# Student Name : ID
# Ahmad Khdair : 1230500
# Malek Zeghari : 1230358

import argparse, math, time, csv, statistics
import numpy as np
import matplotlib.pyplot as plt

LAMBDAS     = [1, 10, 50, 100]
RUNS        = 10            # many times run the code to get:mean cost, best cost, variance, runtime average
SEED        = 7 # to get the same data from the random 

# used in random mode only)
N_DEFAULT   = 100
M_DEFAULT   = 100
SPACE       = 100

HC_MAX_ITER = 500           # HC max iterations
SA_T0       = 1000.0        # SA initial temperature
SA_ALPHA    = 0.95          # SA cooling rate
SA_TMIN     = 1e-10         # SA minimum temperature
EVAL_BUDGET = 20_000        #fair comparison between HC and SA

# Separate seed offsets
SEED_OFFSET = {"Hill Climbing": 0, "Simulated Annealing": 50_000}

def build_distance_matrix(P, C):
    diff = P[:, None, :] - C[None, :, :]   # shape (N, M, 2) (here we get the differnce between the x axis, y axis only, not the dest)
    return np.linalg.norm(diff, axis=2)     # shape (N, M)

def generate_problem():
    rng = np.random.default_rng(SEED)
    P = rng.uniform(0, SPACE, (N_DEFAULT, 2))
    W = rng.integers(1, 11, N_DEFAULT).astype(float)
    C = rng.uniform(0, SPACE, (M_DEFAULT, 2))
    D = build_distance_matrix(P, C)
    return P, W, C, D

def load_problem(pop_file, cand_file):
    # Population CSV: columns x, y, weight
    pop_rows = []
    with open(pop_file, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"x", "y", "weight"} - set(reader.fieldnames or []) #for case if the file is empty the filednames will be None set(None) error so we add or []
        if missing:
            raise ValueError(f"Population file missing columns: {missing}")
        for row in reader:
            pop_rows.append([float(row["x"]), float(row["y"]), float(row["weight"])])
    if not pop_rows:
        raise ValueError("Population file is empty.")
    pop_arr = np.array(pop_rows)
    P = pop_arr[:, :2]
    W = pop_arr[:, 2]

    # Candidates CSV: columns x, y
    cand_rows = []
    with open(cand_file, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"x", "y"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidates file missing columns: {missing}")
        for row in reader:
            cand_rows.append([float(row["x"]), float(row["y"])])
    if not cand_rows:
        raise ValueError("Candidates file is empty.")
    C = np.array(cand_rows)

    D = build_distance_matrix(P, C)
    return P, W, C, D


def random_solution(m, rng, init_rate=None):
    rate = init_rate if init_rate is not None else rng.uniform(0.10, 0.20) # what rate should i take  ? 
    sol  = rng.random(m) < rate
    if not sol.any(): # if the all C are false !!
        sol[rng.integers(m)] = True     # guarantee at least one hospital
    return sol

def evaluate(sol, W, D, lam):
    selected = np.where(sol)[0]#return tuple so i shoould take my list which its at [0]
    if selected.size == 0:
        return float("inf"), 0, float("inf")#No hospital so i should not take this solution 
    min_dist = D[:, selected].min(axis=1) # 
    travel   = float(W @ min_dist)
    h        = int(selected.size) # since its NumPy, its already return int64 but to be sure the everything will work good 
    return travel + lam * h, h, travel / float(W.sum())

def hill_climbing(W, D, lam, rng, eval_budget=EVAL_BUDGET, init_rate=None):
    m    = D.shape[1]
    sol  = random_solution(m, rng, init_rate)
    cost, _, _ = evaluate(sol, W, D, lam)
    evals = 1

    for _ in range(HC_MAX_ITER):
        if evals >= eval_budget:
            break

        best_i, best_cost = None, cost
        for i in range(m):
            if evals >= eval_budget:
                break
            if sol[i] and sol.sum() == 1:       # never remove last hospital
                continue
            nb = sol.copy(); nb[i] = ~nb[i]
            n_cost, _, _ = evaluate(nb, W, D, lam)
            evals += 1
            if n_cost < best_cost:
                best_i, best_cost = i, n_cost

        if best_i is None:                      # local optimum reached
            break
        sol[best_i] = ~sol[best_i]
        cost = best_cost

    final_cost, hospitals, avg_dist = evaluate(sol, W, D, lam)
    return sol, final_cost, hospitals, avg_dist, evals

def simulated_annealing(W, D, lam, rng, eval_budget=EVAL_BUDGET, alpha=SA_ALPHA):
    m    = D.shape[1] #number of loca that i can build hospitals . . 
    sol  = random_solution(m, rng) #out first solution , as inital state . 
    cost, _, _ = evaluate(sol, W, D, lam) # cost of the inital state 
    best_sol, best_cost = sol.copy(), cost #its the inital solutin so its the best till now
    evals = 1 # since i already do calculations for the first solution, so we start from 1 not 0

    if m == 1: # if there is only one option, surly its the solution 
        final_cost, hospitals, avg_dist = evaluate(sol, W, D, lam)
        return sol, final_cost, hospitals, avg_dist, evals

    while evals < eval_budget:
        T = max(SA_T0 * (alpha ** evals), SA_TMIN)

        i = rng.integers(m)
        if sol[i] and sol.sum() == 1:         # never remove last hospital
            continue
        nb = sol.copy();
        nb[i] = ~nb[i]
        n_cost, _, _ = evaluate(nb, W, D, lam)
        evals += 1

        delta = n_cost - cost
        if delta < 0 or rng.random() < math.exp( -delta / T): # if the new solution is good-> take it, else we may take it and may not 
            sol, cost = nb, n_cost
            if cost < best_cost:# we may be in a bad solution then we arrive a good solution, but maybe its not the best solution we arrive from the beginning 
                best_sol, best_cost = sol.copy(), cost

    final_cost, hospitals, avg_dist = evaluate(best_sol, W, D, lam)
    return best_sol, final_cost, hospitals, avg_dist, evals

def run_experiment(name, algorithm, W, D, lam, runs=RUNS, **kwargs):
    rows, best_sol, best_cost = [], None, float("inf")
    offset = SEED_OFFSET.get(name,99_000)

    for r in range(runs):
        rng = np.random.default_rng(SEED + 1000 * r + int(lam) + offset)
        t0  = time.perf_counter()
        sol, cost, hospitals, avg_dist, evals = algorithm(W, D, lam, rng, **kwargs)
        rows.append({
            "algorithm":  name,   "lambda": lam,   "run": r + 1,
            "cost":       cost,   "runtime": time.perf_counter() - t0,
            "hospitals":  hospitals, "avg_travel": avg_dist,
            "evaluations": evals,
        })
        if cost < best_cost:
            best_cost, best_sol = cost, sol.copy()

    return rows, best_sol

def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["algorithm"], r["lambda"]), []).append(r)
    out = []
    for (alg, lam), g in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        costs = [r["cost"] for r in g]
        out.append({
            "algorithm":       alg,  "lambda": lam,
            "mean_cost":       statistics.mean(costs),
            "best_cost":       min(costs),
            "variance":        statistics.variance(costs) if len(costs) > 1 else 0.0,
            "mean_runtime":    statistics.mean(r["runtime"]     for r in g),
            "mean_hospitals":  statistics.mean(r["hospitals"]   for r in g),
            "mean_avg_travel": statistics.mean(r["avg_travel"]  for r in g),
            "mean_evals":      statistics.mean(r["evaluations"] for r in g),
        })
    return out

def print_table(title, rows):
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"{'Algorithm':<26}{'lam':<7}{'MeanCost':<12}{'BestCost':<12}"
          f"{'Variance':<13}{'Hosp':<7}{'AvgTravel':<11}{'Time(s)':<10}{'Evals'}")
    for r in rows:
        print(f"{r['algorithm']:<26}{r['lambda']:<7.0f}{r['mean_cost']:<12.2f}"
              f"{r['best_cost']:<12.2f}{r['variance']:<13.2f}"
              f"{r['mean_hospitals']:<7.1f}{r['mean_avg_travel']:<11.2f}"
              f"{r['mean_runtime']:<10.4f}{r['mean_evals']:.0f}")

def save_csv(filename, rows):
    if not rows: return
    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

def plot_solution(P, C, sol, title, filename):
    hospitals = C[sol]
    plt.figure(figsize=(7, 6))
    plt.scatter(C[:, 0], C[:, 1], s=10, alpha=0.25, c="gray",     label="Candidate sites")
    plt.scatter(P[:, 0], P[:, 1], s=15, c="steelblue",             label="Population")
    plt.scatter(hospitals[:, 0], hospitals[:, 1], s=90, marker="X",
                c="red", label=f"Hospitals ({int(sol.sum())})")
    plt.title(title); plt.xlabel("x"); plt.ylabel("y")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(filename, dpi=150);
    plt.close()

def plot_metric(summary, key, ylabel, filename):
    algs = sorted({r["algorithm"] for r in summary})
    plt.figure(figsize=(7, 5))
    for alg in algs:
        data = sorted([r for r in summary if r["algorithm"] == alg], key=lambda x: x["lambda"])
        plt.plot([r["lambda"] for r in data], [r[key] for r in data], marker="o", label=alg)
    plt.xscale("log"); plt.xlabel("lambda"); plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs lambda"); plt.grid(True, alpha=0.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(filename, dpi=150); plt.close()

def parameter_tuning(W, D):

    lam, rows = 50, []

    for rate in [0.05, 0.15, 0.30]:
        r, _ = run_experiment(
            f"HC init={int(rate*100)}%", hill_climbing,
            W, D, lam, runs=5, init_rate=rate
        )
        rows.extend(r)

    for alpha in [0.90, 0.95, 0.99]:
        r, _ = run_experiment(
            f"SA alpha={alpha}", simulated_annealing,
            W, D, lam, runs=5, alpha=alpha
        )
        rows.extend(r)

    print_table("Parameter Tuning Results (lam=50)", summarize(rows))
    save_csv("parameter_tuning.csv", rows)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population",  type=str, default=None,
                        help="CSV file with columns: x, y, weight  (omit for random mode)")
    parser.add_argument("--candidates",  type=str, default=None,
                        help="CSV file with columns: x, y          (omit for random mode)")
    return parser.parse_args()

def main():
    args = get_args()

    if args.population or args.candidates:
        if not (args.population and args.candidates):
            raise ValueError("Provide both --population and --candidates, or neither.")
        print(f"[File mode] Loading: {args.population}, {args.candidates}")
        P, W, C, D = load_problem(args.population, args.candidates)
    else:
        print("[Random mode] Generating baseline problem (n=100, m=100, seed=7)")
        P, W, C, D = generate_problem()

    n, m = P.shape[0], C.shape[0]
    print(f"Problem: {n} population points, {m} candidate sites")
    print(f"Shared eval budget: {EVAL_BUDGET}  |  Runs: {RUNS}  |  lambda: {LAMBDAS}")

    all_rows = []
    for lam in LAMBDAS:
        for name, algo in [("Hill Climbing",       hill_climbing),
                           ("Simulated Annealing", simulated_annealing)]:
            rows, best = run_experiment(name, algo, W, D, lam)
            all_rows.extend(rows)
            plot_solution(P, C, best, f"{name}  lam={lam}", f"{name}_lambda_{lam}.png")
            
    summary = summarize(all_rows)
    print_table("Main Experiment Results", summary)
    save_csv("all_runs.csv", all_rows)
    save_csv("summary.csv",  summary)

    plot_metric(summary, "mean_cost",       "Mean Total Cost",      "cost_vs_lambda.png")
    plot_metric(summary, "mean_hospitals",  "Mean # Hospitals",     "hospitals_vs_lambda.png")
    plot_metric(summary, "mean_avg_travel", "Mean Avg Travel Dist", "travel_vs_lambda.png")
    plot_metric(summary, "variance",        "Cost Variance",        "variance_vs_lambda.png")

    parameter_tuning(W, D)

    print("\nOutput files saved:")
    if not (args.population or args.candidates):
        print("  population.csv  (generated problem - can be reloaded with --population)")
        print("  candidates.csv  (generated problem - can be reloaded with --candidates)")
    for f in ["all_runs.csv", "summary.csv", "parameter_tuning.csv",
          "cost_vs_lambda.png", "hospitals_vs_lambda.png",
          "travel_vs_lambda.png", "variance_vs_lambda.png"]:
        print(f"  {f}")

    for lam in LAMBDAS:
        print(f"  Hill Climbing_lambda_{lam}.png")
        print(f"  Simulated Annealing_lambda_{lam}.png")

if __name__ == "__main__":
    main()