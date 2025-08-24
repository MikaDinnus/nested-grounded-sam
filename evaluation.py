import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pingouin as pg

# CSV/Excel laden
df = pd.read_excel("evaluation_values.xlsx")

# Spalten umbenennen für Konsistenz
rename_map = {
    "PREC_BOX": "PRECISIONBOX",
    "REC_BOX": "RECALLBOX",
    "F1_BOX": "F1BOX",
    "PREC_SEG": "PRECISIONSEGMENT",
    "REC_SEG": "RECALLSEGMENT",
    "F1_SEG": "F1SEGMENT",
    "MIOU BOX": "MEANIOUBOX",
    "MIOU SEG": "MEANIOUSEGMENT",
}
df.rename(columns=rename_map, inplace=True)

# Crop-Ratio berechnen (Fläche Crop / Fläche Original)
def ratio_from_str(s):
    try:
        w, h = map(int, str(s).split(","))
        return w * h
    except:
        return np.nan

df["CROP_AREA"] = df["CROPPED IMAGE W H"].apply(ratio_from_str)
df["ORG_AREA"] = df["ORG IMAGE W H"].apply(ratio_from_str)
df["CROP_RATIO"] = df["CROP_AREA"] / df["ORG_AREA"]

# Log-transformierte Counts hinzufügen
for col in ["#GT_BOX", "#PRED_BOX", "#GT_SEG", "#PRED_SEG"]:
    df["LOG_" + col] = np.log1p(df[col])

# Nur numerische Spalten für Korrelation
num_df = df.select_dtypes(include=[np.number])

# Korrelationsmatrix (Pearson)
corr = num_df.corr(method="pearson")

plt.figure(figsize=(14,12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Korrelationsmatrix (Nested, inkl. log-Counts & Crop, Pearson)")
plt.tight_layout()
plt.savefig("evaluation_plots/nested_corrmatrix_extended.png", dpi=300)
plt.close()

# Partial Correlations: F1 vs Crop-Ratio kontrolliert für Recall
try:
    part_corr_box = pg.partial_corr(data=df, x="F1BOX", y="CROP_RATIO", covar="RECALLBOX", method="pearson")
    print("Partial Corr (F1BOX ~ CropRatio | RecallBOX):\n", part_corr_box)
except Exception as e:
    print("[WARN] Partial Corr Box fehlgeschlagen:", e)

try:
    part_corr_seg = pg.partial_corr(data=df, x="F1SEGMENT", y="CROP_RATIO", covar="RECALLSEGMENT", method="pearson")
    print("Partial Corr (F1SEGMENT ~ CropRatio | RecallSEGMENT):\n", part_corr_seg)
except Exception as e:
    print("[WARN] Partial Corr Seg fehlgeschlagen:", e)

print("Fertig: Heatmap & Partial-Corr gespeichert.")
