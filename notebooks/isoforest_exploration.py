import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import numpy as np
    import pandas as pd
    import marimo as mo
    import matplotlib.pyplot as plt
    from pathlib import Path
    from scipy.stats import rankdata
    from sklearn.ensemble import IsolationForest
    return IsolationForest, Path, mo, np, pd, plt, rankdata


@app.cell
def __(Path):
    ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    PROC = ROOT / "data" / "processed"
    TAB = ROOT / "reports" / "tables"
    FIG = ROOT / "figures"
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)

    MIN_SIZE = 50
    SEED = 0
    SAMPLE_N = 1_000_000
    RZ = ["log_pay_rz", "log_benes_rz", "srvcs_per_bene_rz", "markup_rz"]
    return FIG, MIN_SIZE, PROC, ROOT, RZ, SAMPLE_N, SEED, TAB


@app.cell
def __(MIN_SIZE, PROC, RZ, np, pd):
    peers = pd.read_parquet(PROC / f"peers_{MIN_SIZE}.parquet")
    _ok = np.isfinite(peers[RZ].to_numpy()).all(axis=1)
    data = peers.loc[_ok].reset_index(drop=True)
    X = data[RZ].to_numpy(dtype="float32")
    return X, data, peers


@app.cell
def __(IsolationForest, SEED, np, rankdata):
    def fit(X_in, seed=SEED, max_samples=256, n_estimators=200):
        return IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                               random_state=seed, n_jobs=-1).fit(X_in)

    def raw_score(model, X_in, chunk=1_000_000):  # higher = more anomalous, chunked
        out = np.empty(len(X_in), dtype="float32")
        for i in range(0, len(X_in), chunk):
            out[i:i + chunk] = -model.score_samples(X_in[i:i + chunk])
        return out 

    def to100(r):
        return (rankdata(r) / len(r) * 100).astype("float32")   # percentile

    def spearman(a, b):
        return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])

    def overlap(a, b, pct):
        k = int(len(a) * pct / 100)
        return len(np.intersect1d(np.argpartition(-a, k)[:k], np.argpartition(-b, k)[:k])) / k
    return fit, overlap, raw_score, spearman, to100


@app.cell
def __(FIG, X, data, fit, np, plt, raw_score, to100):
    _raw = raw_score(fit(X), X)
    res = data.assign(if_raw=_raw, if_score=to100(_raw), base_raw=np.abs(X).max(axis=1))
    res = res.assign(base_score=to100(res["base_raw"].to_numpy()))   # baseline: max |z| across features

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.hist(res["if_raw"], bins=100)
    _ax.set_title("Isolation Forest raw anomaly score (higher = more unusual)")
    _ax.set_xlabel("raw score")
    _ax.set_ylabel("rows")
    _fig.tight_layout()
    _fig.savefig(FIG / "if_raw_hist.png", dpi=150)
    _fig
    return (res,)


@app.cell
def __(TAB, overlap, pd, res, spearman):
    _a, _b = res["if_raw"].to_numpy(), res["base_raw"].to_numpy()
    agree = pd.Series({"spearman": spearman(_a, _b),
                       **{f"top{p}% overlap": overlap(_a, _b, p) for p in (1, 5, 10)}},
                      name="Isolation Forest vs z-score baseline").round(3).to_frame()
    agree.to_csv(TAB / "iforest_vs_baseline.csv")
    return (agree,)


@app.cell
def __(SAMPLE_N, SEED, TAB, X, fit, np, overlap, pd, raw_score, spearman):
    Xs = X[np.random.default_rng(SEED).choice(len(X), size=min(SAMPLE_N, len(X)), replace=False)]
    _seeds = [0, 1, 2, 3, 4]
    _raws = {s: raw_score(fit(Xs, seed=s), Xs) for s in _seeds}
    ref = _raws[0]
    stability = pd.DataFrame(
        [{"pair": f"{a}-{b}", "spearman": spearman(_raws[a], _raws[b]),
          **{f"top{p}%": overlap(_raws[a], _raws[b], p) for p in (1, 5, 10)}}
         for i, a in enumerate(_seeds) for b in _seeds[i + 1:]]).set_index("pair")
    stability.to_csv(TAB / "stability_seeds.csv")
    return Xs, ref, stability


@app.cell
def __(RZ, TAB, Xs, fit, overlap, pd, raw_score, ref, spearman):
    # sensitivity vs the seed-0 reference: tree sample size and dropping one feature at a time
    _runs = ([(f"max_samples={m}", slice(None), {"max_samples": m}) for m in (128, 512)]
             + [(f"drop {c}", [k for k in range(len(RZ)) if k != j], {}) for j, c in enumerate(RZ)])
    _rows = []
    for _name, _cols, _kw in _runs:
        _Xv = Xs[:, _cols]
        _r = raw_score(fit(_Xv, **_kw), _Xv)
        _rows.append({"setting": _name, "spearman": spearman(ref, _r),
                      **{f"top{p}%": overlap(ref, _r, p) for p in (1, 5, 10)}})
    sens = pd.DataFrame(_rows).set_index("setting").round(3)
    sens.to_csv(TAB / "sensitivity_settings.csv")
    return (sens,)


@app.cell
def __(RZ, TAB, np, pd, res):
    res2 = res.assign(dominant=pd.Categorical.from_codes(np.abs(res[RZ].to_numpy()).argmax(axis=1),
                                                         categories=RZ))
    _absz = res2[RZ].abs()
    profile = pd.DataFrame([_absz[res2["if_score"] >= 100 - p].median().rename(f"top {p}%") for p in (1, 5, 10)]
                           + [_absz[res2["if_score"] < 90].median().rename("rest (<90)")]).round(2)
    drivers = pd.DataFrame({
        "all %": res2["dominant"].value_counts(normalize=True) * 100,
        "top 1% %": res2.loc[res2["if_score"] >= 99, "dominant"].value_counts(normalize=True) * 100,
    }).round(1)
    profile.to_csv(TAB / "topk_profile.csv")
    # profile = median |z| by risk group; drivers = which feature has the largest |z|
    return drivers, profile, res2


@app.cell
def __(
    PROC,
    RZ,
    SAMPLE_N,
    SEED,
    TAB,
    data,
    fit,
    np,
    overlap,
    pd,
    raw_score,
    spearman,
):
    _ids = data["row_id"].to_numpy()[
        np.random.default_rng(SEED).choice(len(data), size=min(SAMPLE_N, len(data)), replace=False)]
    _sizes = (30, 50, 100)
    _z = {m: pd.read_parquet(PROC / f"peers_{m}.parquet", columns=["row_id", "peer_level"] + RZ)
          .set_index("row_id").reindex(_ids) for m in _sizes}
    _ok = np.logical_and.reduce([np.isfinite(_z[m][RZ].to_numpy()).all(axis=1) for m in _sizes])

    _raw = {}
    for _m in _sizes:
        _Xm = _z[_m].loc[_ok, RZ].to_numpy(dtype="float32")
        _raw[_m] = raw_score(fit(_Xm), _Xm)

    minsize = pd.DataFrame(
        [{"pair": f"{a} vs {b}", "spearman": spearman(_raw[a], _raw[b]),
          **{f"top{p}%": overlap(_raw[a], _raw[b], p) for p in (1, 5, 10)}}
         for i, a in enumerate(_sizes) for b in _sizes[i + 1:]]).set_index("pair").round(3)
    minsize.to_csv(TAB / "sensitivity_min_size.csv")
    return (minsize,)


@app.cell
def __(TAB, X, fit, overlap, pd, raw_score, res, spearman):
    _top = (res["if_score"] >= 99).to_numpy()
    _rows = []
    for _lvl in (1, 2, 3):
        _mask = (res["peer_level"] == _lvl).to_numpy()
        _Xl = X[_mask]
        _local = raw_score(fit(_Xl), _Xl)
        _glob = res.loc[_mask, "if_raw"].to_numpy()
        _rows.append({"peer_level": _lvl, "rows": int(_mask.sum()),
                      "share of rows %": 100 * _mask.mean(),
                      "share of top 1% %": 100 * (_top & _mask).sum() / _top.sum(),
                      "spearman local vs global": spearman(_local, _glob),
                      "top1% overlap": overlap(_local, _glob, 1)})
    bylevel = pd.DataFrame(_rows).set_index("peer_level").round(3)
    bylevel.to_csv(TAB / "by_peer_level.csv")
    return (bylevel,)


@app.cell
def __(FIG, agree, plt, res):
    # CHART 1: do Isolation Forest and the z-score baseline agree?
    _s = res.sample(min(200_000, len(res)), random_state=0)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _h = _a1.hexbin(_s["base_raw"], _s["if_raw"], gridsize=60, bins="log", cmap="viridis")
    _a1.set_xlabel("Baseline score (largest |z| across features)")
    _a1.set_ylabel("Isolation Forest raw score")
    _a1.set_title(f"Row-by-row agreement (Spearman = {agree.loc['spearman'].iloc[0]:.2f})")
    _fig.colorbar(_h, ax=_a1, label="rows (log scale)")

    _o = agree.drop("spearman").iloc[:, 0]
    _labels = [i.replace(" overlap", "") for i in _o.index]
    _a2.bar(_labels, _o.to_numpy(), color="#534AB7")
    for _i, _v in enumerate(_o.to_numpy()):
        _a2.text(_i, _v + 0.02, f"{_v:.0%}", ha="center")
    _a2.set_ylim(0, 1)
    _a2.set_ylabel("share of the top-K set found by both methods")
    _a2.set_title("Top-K overlap: Isolation Forest vs baseline")
    _fig.tight_layout()
    _fig.savefig(FIG / "if_vs_baseline.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, plt, stability):
    # CHART 2: are the top-ranked rows the same when only the random seed changes?
    _cols = ["top1%", "top5%", "top10%"]
    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.boxplot([stability[c] for c in _cols])
    _ax.set_xticks(range(1, len(_cols) + 1))
    _ax.set_xticklabels(_cols)
    _ax.set_ylim(0, 1)
    _ax.set_ylabel("overlap between two seeds")
    _ax.set_title("Seed stability (10 seed pairs)")
    _fig.tight_layout()
    _fig.savefig(FIG / "stability_seeds.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, sens):
    # CHART 3: how much does the top-ranked set change when settings or features change?
    _ax = sens[["top1%", "top5%", "top10%"]].plot.barh(figsize=(8, 4.5), width=0.8)
    _ax.set_xlim(0, 1)
    _ax.set_xlabel("overlap with the reference run (1.0 = identical top set)")
    _ax.set_ylabel("")
    _ax.set_title("Sensitivity to model settings and dropped features")
    _fig = _ax.figure
    _fig.tight_layout()
    _fig.savefig(FIG / "sensitivity_settings.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, minsize):
    # CHART 4: how much does the peer-group size rule (min_size) change the top-ranked set?
    _ax = minsize[["top1%", "top5%", "top10%"]].plot.bar(figsize=(7, 4), rot=0, width=0.8)
    _ax.set_ylim(0, 1)
    _ax.set_xlabel("min_size pair")
    _ax.set_ylabel("overlap of top-K sets")
    _ax.set_title("Sensitivity to the peer definition (min_size)")
    _fig = _ax.figure
    _fig.tight_layout()
    _fig.savefig(FIG / "sensitivity_min_size.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, bylevel):
    # CHART 5: is any peer level over-represented among the flagged rows?
    _ax = bylevel[["share of rows %", "share of top 1% %"]].plot.bar(figsize=(6.5, 4), rot=0, logy=True)
    _ax.set_xlabel("peer level (1 = specialty+HCPCS+place, 2 = HCPCS+place, 3 = HCPCS)")
    _ax.set_ylabel("% (log scale)")
    _ax.set_title("Share of all rows vs share of top 1%, by peer level")
    _fig = _ax.figure
    _fig.tight_layout()
    _fig.savefig(FIG / "peer_level_share.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, drivers, plt, profile):
    # CHART 6: what do the high-risk groups look like? (median |z| and dominant feature)
    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    _im = _a1.imshow(profile.to_numpy(), cmap="magma_r", aspect="auto")
    _a1.set_xticks(range(profile.shape[1]))
    _a1.set_xticklabels([c.replace("_rz", "") for c in profile.columns], rotation=20)
    _a1.set_yticks(range(profile.shape[0]))
    _a1.set_yticklabels(profile.index)
    for _i in range(profile.shape[0]):
        for _j in range(profile.shape[1]):
            _val = profile.iloc[_i, _j]
            _a1.text(_j, _i, f"{_val:.2f}", ha="center", va="center",
                     color="white" if _im.norm(_val) > 0.5 else "black")
    _a1.set_title("Median |z| by risk group (bigger = further from peers)")
    _fig.colorbar(_im, ax=_a1)

    drivers.rename(index=lambda s: s.replace("_rz", "")).plot.bar(ax=_a2, rot=20)
    _a2.set_ylabel("% of rows")
    _a2.set_xlabel("")
    _a2.set_ylim(0, _a2.get_ylim()[1] * 1.25)
    _a2.set_title("Which feature is most extreme? (dominant driver)")
    _fig.tight_layout()
    _fig.savefig(FIG / "risk_group_profile.png", dpi=150)
    _fig
    return


@app.cell
def __(FIG, PROC, pd, plt, res2):
    # CHART 7: which specialties and HCPCS codes are flagged more often than the 1% you would expect?
    _meta = pd.read_parquet(PROC / "features.parquet",
                            columns=["row_id", "Rndrng_Prvdr_Type", "HCPCS_Cd"])
    _top = res2[res2["if_score"] >= 99].merge(_meta, on="row_id")

    def _rate(col, min_rows):
        _tot = _meta[col].value_counts()
        _hit = _top[col].value_counts()
        _r = (_hit.reindex(_tot.index).fillna(0) / _tot * 100).where(_tot >= min_rows).dropna()
        return _r.sort_values().tail(10)

    _fig, (_a1, _a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _rate("Rndrng_Prvdr_Type", 1000).plot.barh(ax=_a1, color="#0F6E56")
    _a1.set_title("Specialties: % of rows in the top 1% (expected: 1%)")
    _rate("HCPCS_Cd", 500).plot.barh(ax=_a2, color="#0F6E56")
    _a2.set_title("HCPCS codes: % of rows in the top 1% (expected: 1%)")
    for _a in (_a1, _a2):
        _a.axvline(1, color="red", linestyle="--")
        _a.set_ylabel("")
        _a.set_xlabel("% of that group's rows flagged")
    _fig.tight_layout()
    _fig.savefig(FIG / "top1_by_group.png", dpi=150)
    _fig
    return


@app.cell
def __(TAB, agree, data, minsize, pd, peers, sens, stability):
    # headline numbers for the write-up
    summary = pd.Series({
        "rows loaded": f"{len(peers):,}",
        "rows modeled": f"{len(data):,}",
        "IF vs baseline: Spearman": round(float(agree.loc["spearman"].iloc[0]), 3),
        "IF vs baseline: top 1% overlap": round(float(agree.loc["top1% overlap"].iloc[0]), 3),
        "seed stability: mean top 1% overlap": round(float(stability["top1%"].mean()), 3),
        "seed stability: worst top 1% overlap": round(float(stability["top1%"].min()), 3),
        "settings/features: lowest top 1% overlap": round(float(sens["top1%"].min()), 3),
        "min_size: lowest top 1% overlap": round(float(minsize["top1%"].min()), 3),
    }, name="value", dtype=object).to_frame()
    summary.to_csv(TAB / "headline_numbers.csv")
    summary
    return (summary,)


@app.cell
def __(PROC, res2):
    res2[["row_id", "if_score", "base_score", "dominant"]].to_parquet(PROC / "isoforest_scores.parquet", index=False)
    return


if __name__ == "__main__":
    app.run()
