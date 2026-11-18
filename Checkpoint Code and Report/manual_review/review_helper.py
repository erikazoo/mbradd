import sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "manual_review_sample.csv"
df = pd.read_csv(path, dtype={"HCPCS_Cd": str, "Rndrng_NPI": str})

FEATS = ["std. payment ($)", "patients", "services per patient", "charge / allowed"]

for f in FEATS:
    x, med, z = df[f], df[f"{f} peer median"], df[f"{f} z"]
    # how far from the median in plain terms (+0.05 = 5% above)
    df[f"{f} rel diff"] = (x / med - 1).round(3)
    # raw-unit distance that equals ONE z unit in this peer group (rough: z is on transformed features)
    df[f"{f} 1z ="] = ((x - med) / z).replace([np.inf, -np.inf], np.nan).abs().round(4)
    # big z but small real difference -> the z is inflated by a near-zero spread (tiny MAD)
    df[f"{f} tiny-MAD?"] = (z.abs() > 3.5) & ((x / med - 1).abs() < 0.15)

# approximate dollars at stake on this row: patients x services per patient x payment per service
df["approx services"] = (df["patients"] * df["services per patient"]).round(0)
df["approx total paid ($)"] = (df["approx services"] * df["std. payment ($)"]).round(0)

# which features are really (not just statistically) different
def real_drivers(r):
    out = []
    for f in FEATS:
        z, rel = r[f"{f} z"], r[f"{f} rel diff"]
        if pd.notna(z) and abs(z) > 3.5 and abs(rel) >= 0.15:
            out.append(f"{f} {'high' if z > 0 else 'low'} ({r[f + ' x median']}x, z={z})")
    return "; ".join(out)
df["real drivers"] = df.apply(real_drivers, axis=1)
df["artifact flags"] = df.apply(
    lambda r: ", ".join(f for f in FEATS if r[f"{f} tiny-MAD?"]), axis=1)

view = df[["review_group", "row_id", "HCPCS_Cd", "Rndrng_Prvdr_Type", "peer_level",
           "real drivers", "artifact flags", "approx services", "approx total paid ($)"]]
pd.set_option("display.width", 250, "display.max_colwidth", 90, "display.max_rows", 100)
print(view.to_string(index=False))
print("\nrows per group:\n", df["review_group"].value_counts().to_string())
df.to_csv("manual_review_with_helpers.csv", index=False)