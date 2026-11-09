import os
import numpy as np
import pandas as pd

FEATS = ["log_pay", "log_benes", "srvcs_per_bene", "markup"]
RZ_COLS = [f + "_rz" for f in FEATS]
LEVELS = {1: ["Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc"],
          2: ["HCPCS_Cd", "Place_Of_Srvc"],
          3: ["HCPCS_Cd"]}


def load_clean(raw_path):
    cols = ["Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc", "Rndrng_Prvdr_Cntry",
            "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs",
            "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Stdzd_Amt"]
    df = pd.read_parquet(raw_path, columns=cols)

    # remove logically invalid rows
    keep = ((df["Rndrng_Prvdr_Cntry"] == "US")
            & (df["Tot_Srvcs"] >= df["Tot_Benes"])
            & (df["Tot_Bene_Day_Srvcs"] <= df["Tot_Srvcs"])
            & (df["Avg_Mdcr_Alowd_Amt"] > 0))
    df = df[keep].reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    return df


def add_features(df):
    df["log_pay"] = np.log(df["Avg_Mdcr_Stdzd_Amt"].clip(lower=1e-6))  # payment level (avoid log(0))
    df["log_benes"] = np.log(df["Tot_Benes"])  # volume
    df["srvcs_per_bene"] = np.log(df["Tot_Srvcs"] / df["Tot_Benes"])  # intensity
    df["markup"] = np.log(df["Avg_Sbmtd_Chrg"].clip(lower=1e-6) / df["Avg_Mdcr_Alowd_Amt"])  # charge vs allowed
    return df


def add_groups(df):
    for lvl, keys in LEVELS.items():
        df[f"gid{lvl}"] = df.groupby(keys, observed=True).ngroup() + lvl * 10_000_000  # group id, 10M apart per level
        df[f"size{lvl}"] = df.groupby(keys, observed=True)["row_id"].transform("size")  # rows in the group
    return df


def robust_z(df, keys):
    med = df.groupby(keys, observed=True)[FEATS].transform("median")  # group median
    dev = (df[FEATS] - med).abs()  # distance from the median
    dev[keys] = df[keys]  # add group columns so we can group the deviations
    mad = dev.groupby(keys, observed=True)[FEATS].transform("median")
    mean_dev = dev.groupby(keys, observed=True)[FEATS].transform("mean")
    scale = (1.4826 * mad).where(mad > 0, 1.2533 * mean_dev)  # if MAD = 0, use mean abs deviation
    return ((df[FEATS] - med) / scale.where(scale > 0)).astype("float32")


def make_peers(df, rz_levels, min_size):
    p = df[["row_id"]].copy()
    p["peer_level"] = 0  # 0 = unscored
    p["peer_id"] = -1
    p["peer_n"] = 0
    for lvl in [1, 2, 3]:
        take = (p["peer_level"] == 0) & (df[f"size{lvl}"] >= min_size)  # not assigned yet + group is big enough
        p.loc[take, "peer_level"] = lvl
        p.loc[take, "peer_id"] = df.loc[take, f"gid{lvl}"]
        p.loc[take, "peer_n"] = df.loc[take, f"size{lvl}"]

    p[RZ_COLS] = np.nan  # unscored rows stay NaN
    for lvl in [1, 2, 3]:
        rows = p["peer_level"] == lvl
        p.loc[rows, RZ_COLS] = rz_levels[lvl].loc[rows].to_numpy()  # z-scores from the chosen level
    p[RZ_COLS] = p[RZ_COLS].astype("float32")
    return p


def build_common_setup(raw_path="medicare_2024.parquet", out_dir="data/processed", sizes=(30, 50, 100)):
    os.makedirs(out_dir, exist_ok=True)
    df = add_groups(add_features(load_clean(raw_path)))

    keep_cols = ["row_id", "Rndrng_NPI", "Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc"] + FEATS
    df[keep_cols].to_parquet(os.path.join(out_dir, "features.parquet"), index=False)

    rz_levels = {lvl: robust_z(df, keys) for lvl, keys in LEVELS.items()}  # z-scores if every row used this level
    for m in sizes:
        p = make_peers(df, rz_levels, m)
        p.to_parquet(os.path.join(out_dir, f"peers_{m}.parquet"), index=False)
        levels = p["peer_level"].value_counts(normalize=True).mul(100).round(2).to_dict()
        print(f"min size {m}: % of rows per level {levels}")
    print(f"rows: {len(df):,}, files saved to {out_dir}")


if __name__ == "__main__":
    build_common_setup()
