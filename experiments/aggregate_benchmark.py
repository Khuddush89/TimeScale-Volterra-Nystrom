"""Aggregate the seeded neural benchmark (exp4_neural + exp4_neural_lm) into mean/std/median per method family;
also compare with a reference results directory if given.  Writes results/csv/table_benchmark_*.csv"""
import json, os, sys, re
import numpy as np, pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
def load(d):
    rows = []
    for n in ["exp4_neural", "exp4_neural_lm"]:
        p = os.path.join(d, n + ".json")
        if os.path.exists(p): rows += json.load(open(p))["rows"]
    df = pd.DataFrame(rows); df["label"] = df["method"].str.replace(r" \(seed \d+\)", "", regex=True); return df
df = load(R)
os.makedirs(os.path.join(R, "csv"), exist_ok=True)
df.to_csv(os.path.join(R, "csv", "table_benchmark_runs.csv"), index=False)
g = df.groupby("label", sort=False).agg(runs=("err", "size"), n_params=("n_params", "first"),
    err_median=("err", "median"), err_mean=("err", "mean"), err_std=("err", "std"),
    eta_maj_median=("eta_maj", "median"), eff_maj_mean=("eff_maj", "mean"), eff_maj_std=("eff_maj", "std"),
    time_mean=("time", "mean"), time_std=("time", "std"))
g.to_csv(os.path.join(R, "csv", "table_benchmark_summary.csv"))
pd.set_option("display.width", 250); print(g.to_string(float_format=lambda v: f"{v:.3g}"))
c = json.load(open(os.path.join(R, "exp4_classical.json")))["classical"]
pd.DataFrame(c).to_csv(os.path.join(R, "csv", "table_benchmark_classical.csv"), index=False); print(pd.DataFrame(c).to_string())
if len(sys.argv) > 1:
    o = load(sys.argv[1]).groupby("label", sort=False)["err"].median(); print("\nORIGINAL medians vs rerun:")
    print(pd.DataFrame({"orig": o, "rerun": g["err_median"]}).to_string(float_format=lambda v: f"{v:.3g}"))
