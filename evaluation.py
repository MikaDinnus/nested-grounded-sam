import pandas as pd
from scipy.stats import wilcoxon

# Datei einlesen (fester Header wie von dir angegeben)
df = pd.read_excel("evaluation_backup.xlsx")

# Relevante Metrikspalten exakt nach deinem Header
metrics = [
    "PREC_BOX", "REC_BOX", "F1_BOX",
    "PREC_SEG", "REC_SEG", "F1_SEG",
    "MIOU BOX", "MIOU SEG"
]

# TYPE vereinheitlichen
df["TYPE"] = df["TYPE"].astype(str).str.upper().str.strip()

# Falls es mehrere Zeilen je (CODE, TYPE) gibt: auf Mittelwert aggregieren
agg = df.groupby(["CODE", "TYPE"], as_index=False)[metrics].mean(numeric_only=True)

# Wide-Format: pro CODE je eine Spalte FLAT/NESTED für jede Metrik
wide = agg.pivot(index="CODE", columns="TYPE", values=metrics)

print("\nWilcoxon Signed-Rank Tests (Flat vs Nested):\n")
for m in metrics:
    # Wertepaare (nur Codes, die FLAT und NESTED haben)
    a = wide[m].get("FLAT")
    b = wide[m].get("NESTED")
    if a is None or b is None:
        print(f"{m:15s} keine FLAT/NESTED-Paare gefunden"); continue
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        print(f"{m:15s} keine Paare nach NaN-Filter"); continue

    stat, p = wilcoxon(a[mask].values, b[mask].values)
    print(f"{m:15s} W={stat:.4f}, p={p:.4f}, n={mask.sum()}")
