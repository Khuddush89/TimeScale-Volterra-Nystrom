"""Re-evaluate every saved P5 checkpoint without retraining (review item R2).

For each results/checkpoints/*.npz the trial function is rebuilt from its stored specification, the
stored parameters are loaded, and the sampled error on E (h=2^-10) and the certified bounds are recomputed
and compared with the values recorded at training time.  Writes results/csv/checkpoint_reevaluation.csv."""
from exp4_benchmark import *
import glob, pandas as pd


def run():
    rows = []
    for f in sorted(glob.glob(os.path.join(CKPT, "*.npz"))):
        tr, p, meta = checkpoints.load(f, T, xa)
        vf = tr.numpy_fn(p)
        err = error_fn(vf); c = certify_fn(vf)
        ex = meta["extra"]
        rows.append(dict(checkpoint=os.path.basename(f), method=ex["method"], n_params=meta["n_params"],
                         err_recorded=ex["err"], err_reloaded=err, rel_diff_err=abs(err / ex["err"] - 1.0),
                         eta_maj_recorded=ex["eta_maj"], eta_maj_reloaded=c["eta_maj"],
                         rel_diff_eta=abs(c["eta_maj"] / ex["eta_maj"] - 1.0)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "csv", "checkpoint_reevaluation.csv"), index=False)
    print(df[["method", "err_recorded", "err_reloaded", "rel_diff_err", "rel_diff_eta"]].to_string())
    print("checkpoints:", len(df), " max rel diff err:", df.rel_diff_err.max(), " eta:", df.rel_diff_eta.max())


if __name__ == "__main__":
    run()
