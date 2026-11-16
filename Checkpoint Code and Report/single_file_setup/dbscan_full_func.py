import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

def main():
    df = pd.read_parquet("../data/features.parquet", engine="pyarrow")
    df = df.drop(columns='row_id')
    col_group_1 = [
        "HCPCS_Cd",  # 1
        "Place_Of_Srvc",  # 2
        "Rndrng_Prvdr_Type",  # 3
    ]

    col_group_2 = [
        "log_pay",
        "log_benes",
        "srvcs_per_bene",
        "markup"
    ]
    results = []
    min_group_size = 100
    max_group_size = 999999999

    min_samples = 10
    eps = 0.9

    output_dir = Path.cwd() / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = df.groupby(col_group_1, observed=True, dropna=False)
    total_groups = groups.ngroups

    for i, (group_key, subset) in enumerate(groups, start=1):
        print(f"Group {i:,}/{total_groups:,}", end="\r", flush=True)

        if not min_group_size <= len(subset) <= max_group_size:
            continue
        if not min_group_size <= len(subset) <= max_group_size:
            continue

        subset = subset.copy()
        subset[col_group_2] = (
            subset[col_group_2]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        valid = (
                subset[col_group_2].notna().all(axis=1)
                & subset[col_group_2].ge(0).all(axis=1)
        )
        subset = subset.loc[valid].copy()

        if subset.empty:
            continue

        X = RobustScaler().fit_transform(subset[col_group_2])

        subset["cluster"] = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1,
        ).fit_predict(X)

        subset["neighbor_distance"] = np.nan

        if len(X) >= min_samples:
            neighbors = NearestNeighbors(
                n_neighbors=min_samples,
                n_jobs=-1,
            ).fit(X)

            distances, _ = neighbors.kneighbors(X)
            subset["neighbor_distance"] = distances[:, -1]

        results.append(subset)

    all_results = (
        pd.concat(results, ignore_index=True)
        if results
        else df.iloc[:0].assign(
            cluster=pd.Series(dtype=int),
            neighbor_distance=pd.Series(dtype=float),
        )
    )

    all_results = all_results.sort_values(
        "neighbor_distance",
        ascending=False,
        na_position="last",
        ignore_index=True,
    )

    all_results.to_csv(output_dir / "dbscan_results.csv", index=False)

if __name__ == "__main__":
    main()