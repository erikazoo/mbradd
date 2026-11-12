import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


INPUT_OPTIONS = ["data/features.parquet"]

OUTPUT_CSV = "isolation_forest_full_anomaly_ranking.csv"
MIN_PEER_SIZE = 100

MODEL_FEATURES = [
    "log_pay",
    "log_benes",
    "srvcs_per_bene",
    "markup"
]

PEER_LEVELS = {
    1: ["Rndrng_Prvdr_Type", "HCPCS_Cd", "Place_Of_Srvc"],
    2: ["Place_Of_Srvc", "HCPCS_Cd"],
    3: ["HCPCS_Cd"]
}

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "max_features": 1.0,
    "contamination": "auto",
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1
}


def find_input_file():
    for path in INPUT_OPTIONS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find features.parquet. "
        "Put the script next to the parquet file or update INPUT_OPTIONS."
    )


def add_peer_fallback_info(df_model):

    for level, cols in PEER_LEVELS.items():
        grouped = df_model.groupby(cols, observed=True, sort=False)

        df_model[f"_peer_size_L{level}"] = grouped["row_id"].transform("size")

        df_model[f"_peer_id_L{level}"] = (
            grouped.ngroup().astype("int64") + level * 10_000_000
        )

    df_model["peer_level"] = 0
    df_model["peer_n"] = 0
    df_model["peer_model_id"] = -1

    for level in [1, 2, 3]:

        use_level = (
            (df_model["peer_level"] == 0)
            & (df_model[f"_peer_size_L{level}"] >= MIN_PEER_SIZE)
        )

        df_model.loc[use_level, "peer_level"] = level

        df_model.loc[use_level, "peer_n"] = df_model.loc[
            use_level, f"_peer_size_L{level}"
        ]

        df_model.loc[use_level, "peer_model_id"] = (
            df_model.loc[use_level, f"_peer_id_L{level}"]
        )

    df_model["peer_level"] = df_model["peer_level"].astype("int8")
    df_model["peer_n"] = df_model["peer_n"].astype("int32")

    df_model["peer_model_id"] = (
        df_model["peer_model_id"].astype("int64")
    )

    return df_model


def score_one_level(df_model, level):
    cols = PEER_LEVELS[level]

    number_of_models = (
        df_model.loc[df_model["peer_level"] == level]
        .groupby(cols, observed=True)
        .ngroups
    )

    if number_of_models == 0:
        print(f"Level {level}: no groups to score")
        return

    print(f"\nLevel {level}: scoring {number_of_models:,} peer models")

    scored_models = 0
    start_level = time.perf_counter()

    for _, group in df_model.groupby(cols, observed=True, sort=False):

        target_index = group.index[group["peer_level"] == level]

        if len(target_index) == 0:
            continue

        X = group[MODEL_FEATURES]

        model = IsolationForest(**MODEL_PARAMS)
        model.fit(X)

        raw_scores = pd.Series(
            -model.score_samples(X),
            index=group.index
        )

        peer_percentiles = (
            raw_scores.rank(method="average", pct=True) * 100
        )

        df_model.loc[target_index, "iforest_raw_score"] = (
            raw_scores.loc[target_index].to_numpy()
        )

        df_model.loc[target_index, "anomaly_score"] = peer_percentiles.loc[
            target_index
        ].to_numpy()

        scored_models += 1

        if scored_models % 100 == 0 or scored_models == number_of_models:
            elapsed = time.perf_counter() - start_level

            print(
                f"Finished {scored_models:,}/{number_of_models:,} "
                f"models ({elapsed:.1f}s)"
            )


def main():

    input_path = find_input_file()
    print("Loading:", input_path)

    df = pd.read_parquet(input_path)

    required_cols = (
        ["row_id", "Rndrng_NPI"]
        + MODEL_FEATURES
        + list({
            col
            for cols in PEER_LEVELS.values()
            for col in cols
        })
    )

    missing_cols = [
        col for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df["row_id"].duplicated().any():
        raise ValueError("row_id must be unique.")

    scoreable = (
        df[MODEL_FEATURES + PEER_LEVELS[1]]
        .notna()
        .all(axis=1)
    )

    df_model = df.loc[scoreable].copy()

    print("Rows in file:", f"{len(df):,}")
    print("Rows with complete model data:", f"{len(df_model):,}")

    df_model = add_peer_fallback_info(df_model)

    coverage = (
        df_model["peer_level"]
        .value_counts(normalize=True)
        .sort_index() * 100
    )

    print("\nPeer-level coverage:")

    for level in [0, 1, 2, 3]:
        print(f"Level {level}: {coverage.get(level, 0):.4f}%")

    df_model["iforest_raw_score"] = np.nan
    df_model["anomaly_score"] = np.nan

    for level in [1, 2, 3]:
        score_one_level(df_model, level)

    score_cols = df_model[
        [
            "row_id",
            "peer_level",
            "peer_n",
            "peer_model_id",
            "iforest_raw_score",
            "anomaly_score"
        ]
    ].copy()

    output = df.merge(
        score_cols,
        on="row_id",
        how="left",
        validate="one_to_one"
    )

    output["peer_level"] = output["peer_level"].fillna(0).astype("int8")

    output["peer_n"] = (
        output["peer_n"].fillna(0).astype("int32")
    )

    output["peer_model_id"] = output["peer_model_id"].fillna(-1).astype("int64")

    output["scored"] = output["anomaly_score"].notna()

    output["unscored_reason"] = ""

    missing_model_data = ~scoreable

    output.loc[
        missing_model_data,
        "unscored_reason"
    ] = "missing model feature or peer value"

    too_small = (
        scoreable
        & (output["peer_level"] == 0)
    )

    output.loc[
        too_small,
        "unscored_reason"
    ] = "peer group below 100 at all fallback levels"

    output = output.sort_values(
        by=["anomaly_score", "iforest_raw_score"],
        ascending=[False, False],
        na_position="last",
        kind="mergesort"
    ).reset_index(drop=True)

    output["global_anomaly_rank"] = pd.Series(
        pd.NA,
        index=output.index,
        dtype="Int64"
    )

    scored_mask = output["scored"]
    n_scored = int(scored_mask.sum())

    output.loc[
        scored_mask,
        "global_anomaly_rank"
    ] = np.arange(1, n_scored + 1)

    first_cols = [
        "global_anomaly_rank",
        "anomaly_score",
        "iforest_raw_score",
        "scored",
        "peer_level",
        "peer_n",
        "peer_model_id"
    ]

    remaining_cols = [
        col for col in output.columns
        if col not in first_cols
    ]

    output = output[first_cols + remaining_cols]

    print("\nScored rows:", f"{n_scored:,}")

    print(
        "Unscored rows:",
        f"{len(output) - n_scored:,}"
    )

    print("\nTop 10 anomalies:")

    print(
        output[
            [
                "global_anomaly_rank",
                "Rndrng_NPI",
                "Rndrng_Prvdr_Type",
                "HCPCS_Cd",
                "Place_Of_Srvc",
                "peer_level",
                "peer_n",
                "anomaly_score",
                "iforest_raw_score"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nWriting CSV:", OUTPUT_CSV)

    output.to_csv(OUTPUT_CSV, index=False)

    print("Done.")

    print(
        "anomaly_score is a 0-100 "
        "peer-relative percentile, "
        "not a fraud probability."
    )


if __name__ == "__main__":
    main()