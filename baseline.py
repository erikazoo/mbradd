import os
import numpy as np
import pandas as pd

RZ_COLS = ["log_pay_rz", "log_benes_rz", "srvcs_per_bene_rz", "markup_rz"]


def run_baseline(peers_path="data/processed/peers_100.parquet",
                 out_path="data/processed/baseline_output.parquet",
                 one=10, two=3.5):
    p = pd.read_parquet(peers_path)
    scored = p["peer_level"] > 0
    high = p[RZ_COLS].clip(lower=0).fillna(0)  # only above peers

    z_sorted = np.sort(high.to_numpy(), axis=1)  # each row sorted ascending
    top1, top2 = z_sorted[:, -1], z_sorted[:, -2]  # largest and 2nd largest z
    # 1 feature > one OR 2+ features > two  <=>  raw > two
    raw = pd.Series(np.maximum(top2, top1 * two / one), index=p.index).where(scored)

    out = p[["row_id", "peer_level"]].copy()
    out.insert(1, "method", "baseline_zscore")
    out["score"] = (raw.rank(pct=True) * 100).round(1)
    out["flag"] = raw > two

    fl = out["flag"]
    out["risk_tier"] = "None"
    out.loc[fl, "risk_tier"] = pd.cut(raw[fl].rank(pct=True), [0, 0.5, 0.85, 1],
                                      labels=["Low", "Medium", "High"]).astype(str)

    names = [c.replace("_rz", "") for c in RZ_COLS]
    over = (high > two).to_numpy()
    out["reason"] = [", ".join(n for n, m in zip(names, row) if m) for row in over]
    out.loc[~fl, "reason"] = ""  # reason only for flagged rows

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"flagged: {fl.sum():,} rows ({fl.mean() * 100:.2f}%), saved to {out_path}")
    return out


if __name__ == "__main__":
    run_baseline()
