"""
EE244 Computational Learning — PA #1
Decision Trees & Random Forest
Datasets: Mushroom (parts a-c), Loan Prediction (part d)
Author: Kamren
Due: April 23, 2026
"""

# ─────────────────────────────────────────────
# 0. Imports & Setup
# ─────────────────────────────────────────────
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def encode_categorical(df):
    """Label-encode all object columns in-place and return the dataframe."""
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def impute_df(df):
    """Fill missing values with column mode (safe for categorical datasets)."""
    imp = SimpleImputer(strategy="most_frequent")
    df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# PART (a) — Baseline Decision Tree on Mushroom Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART (a) — Baseline Decision Tree | Mushroom Dataset")
print("=" * 65)

# ---- Load & preprocess ----
# Dataset: https://www.kaggle.com/datasets/uciml/mushroom-classification
# Assumes file is downloaded as 'mushrooms.csv' in the working directory.
# The UCI mushroom dataset has 22 categorical features + 1 label column ('class').
try:
    mush = pd.read_csv("datasets/mushrooms/mushroom.csv")
except FileNotFoundError:
    # Fallback: synthesise a small representative dataset so the script
    # still demonstrates the full pipeline even without the Kaggle file.
    print("[INFO] mushrooms.csv not found — generating synthetic stand-in (500 rows).")
    rng = np.random.default_rng(0)
    n = 500
    feat_vals = {"cap-shape": list("bcxfks"),
                 "cap-color": list("nbcgryuwpe"),
                 "odor":      list("alcyfmnps"),
                 "gill-size": ["broad", "narrow"],
                 "spore-color": list("nbhrouywr"),
                 "population": list("abcnsvyw"),
                 "habitat":   list("dghlmpuw")}
    rows = {k: rng.choice(v, n) for k, v in feat_vals.items()}
    # synthetic rule: 'p' odor → poisonous, else edible (noisy)
    label = np.where(rows["odor"] == "p", "p", "e")
    flip = rng.random(n) < 0.05
    label[flip] = np.where(label[flip] == "p", "e", "p")
    rows["class"] = label
    mush = pd.DataFrame(rows)

print(f"\nDataset shape : {mush.shape}")
print(f"Target counts :\n{mush['class'].value_counts().to_string()}\n")

mush = encode_categorical(mush)
mush = impute_df(mush)

X_mush = mush.drop(columns=["class"])
y_mush = mush["class"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_mush, y_mush, test_size=0.2, random_state=42, stratify=y_mush
)

# ---- Fit baseline (no pruning) ----
dt_base = DecisionTreeClassifier(criterion="entropy", random_state=42)
dt_base.fit(X_train_m, y_train_m)

train_err_base = 1 - accuracy_score(y_train_m, dt_base.predict(X_train_m))
test_err_base  = 1 - accuracy_score(y_test_m,  dt_base.predict(X_test_m))
print(f"Baseline DT  — train error: {train_err_base:.4f}  |  test error: {test_err_base:.4f}")
print(f"Tree depth   : {dt_base.get_depth()}   |  leaves: {dt_base.get_n_leaves()}")
print(f"\nClassification Report (test set):\n{classification_report(y_test_m, dt_base.predict(X_test_m))}")

# ─────────────────────────────────────────────────────────────────────────────
# PART (b) — Pruning & Comparison with k-NN
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("PART (b) — Pruning Experiments + k-NN Comparison")
print("=" * 65)

# We implement "cost-complexity pruning" via ccp_alpha (sklearn's built-in
# reduced-error pruning equivalent) AND max_depth pruning for illustration.

# ---- max_depth pruning sweep ----
depths       = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
depth_labels = [str(d) if d else "None" for d in depths]
train_errs_d, test_errs_d = [], []

for d in depths:
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=42)
    clf.fit(X_train_m, y_train_m)
    train_errs_d.append(1 - accuracy_score(y_train_m, clf.predict(X_train_m)))
    test_errs_d.append (1 - accuracy_score(y_test_m,  clf.predict(X_test_m)))
    print(f"  max_depth={str(d):>4s}  |  train err={train_errs_d[-1]:.4f}  |  test err={test_errs_d[-1]:.4f}")

# ---- ccp_alpha pruning sweep (reduced-error style) ----
path   = dt_base.cost_complexity_pruning_path(X_train_m, y_train_m)
alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas)//15)]  # subsample 15 pts

train_errs_a, test_errs_a, used_alphas = [], [], []
for a in alphas:
    clf = DecisionTreeClassifier(criterion="entropy", ccp_alpha=a, random_state=42)
    clf.fit(X_train_m, y_train_m)
    if clf.get_n_leaves() < 2:
        break
    train_errs_a.append(1 - accuracy_score(y_train_m, clf.predict(X_train_m)))
    test_errs_a.append (1 - accuracy_score(y_test_m,  clf.predict(X_test_m)))
    used_alphas.append(a)

# ---- k-NN comparison ----
print("\nk-NN sweep:")
k_vals, knn_test_errs = [1, 3, 5, 7, 11, 15, 21], []
for k in k_vals:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_m, y_train_m)
    err = 1 - accuracy_score(y_test_m, knn.predict(X_test_m))
    knn_test_errs.append(err)
    print(f"  k={k:>2d}  test err={err:.4f}")

best_k = k_vals[np.argmin(knn_test_errs)]
best_knn_err = min(knn_test_errs)
print(f"\nBest k-NN: k={best_k}, test error={best_knn_err:.4f}")

# ---- Plot part (b) ----
fig_b, axes_b = plt.subplots(1, 2, figsize=(13, 5))
fig_b.suptitle("Part (b) — Pruning Effects & k-NN Comparison | Mushroom Dataset", fontsize=13)

ax1 = axes_b[0]
ax1.plot(depth_labels, train_errs_d, "o-", label="Train error", color="steelblue")
ax1.plot(depth_labels, test_errs_d,  "s--", label="Test error",  color="tomato")
ax1.axhline(best_knn_err, linestyle=":", color="seagreen", label=f"Best k-NN (k={best_k})")
ax1.set_xlabel("max_depth (None = unpruned)")
ax1.set_ylabel("Error rate")
ax1.set_title("Pruning via max_depth")
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes_b[1]
ax2.plot(used_alphas, train_errs_a, "o-", label="Train error", color="steelblue")
ax2.plot(used_alphas, test_errs_a,  "s--", label="Test error",  color="tomato")
ax2.axhline(best_knn_err, linestyle=":", color="seagreen", label=f"Best k-NN (k={best_k})")
ax2.set_xlabel("ccp_alpha (cost-complexity)")
ax2.set_ylabel("Error rate")
ax2.set_title("Pruning via ccp_alpha")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("part_b_pruning.png", dpi=150)
plt.show()
print("\n[Saved] part_b_pruning.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART (c) — Learning Curves (time & error vs. training set size)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART (c) — Learning Curves | Mushroom Dataset")
print("=" * 65)

# Hold out 30% for test; vary training size over 10%–70% of the remaining 70%
X_tr70, X_test_c, y_tr70, y_test_c = train_test_split(
    X_mush, y_mush, test_size=0.30, random_state=42, stratify=y_mush
)

fracs      = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
n_examples, train_times, test_errors_c = [], [], []

for frac in fracs:
    n = max(2, int(frac * len(X_tr70)))
    X_sub = X_tr70.iloc[:n]
    y_sub = y_tr70.iloc[:n]

    t0 = time.perf_counter()
    clf_c = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf_c.fit(X_sub, y_sub)
    elapsed = time.perf_counter() - t0

    err = 1 - accuracy_score(y_test_c, clf_c.predict(X_test_c))
    n_examples.append(n)
    train_times.append(elapsed * 1000)   # ms
    test_errors_c.append(err)
    print(f"  {frac*100:>3.0f}%  n={n:>5d}  time={elapsed*1000:.3f} ms  test_err={err:.4f}")

# ---- Plot part (c) ----
fig_c, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(13, 5))
fig_c.suptitle("Part (c) — Learning Curves | Mushroom Dataset", fontsize=13)

ax_t.plot(n_examples, train_times, "o-", color="darkorange")
ax_t.set_xlabel("Number of training examples")
ax_t.set_ylabel("Induction time (ms)")
ax_t.set_title("Computation time vs. training size")
ax_t.grid(alpha=0.3)

ax_e.plot(n_examples, test_errors_c, "s-", color="tomato")
ax_e.set_xlabel("Number of training examples")
ax_e.set_ylabel("Test error rate")
ax_e.set_title("Test error rate vs. training size")
ax_e.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("part_c_learning_curves.png", dpi=150)
plt.show()
print("[Saved] part_c_learning_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# PART (d) — Random Forest | Loan Prediction Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PART (d) — Random Forest | Loan Prediction Dataset")
print("=" * 65)

try:
    loan = pd.read_csv("datasets/loans/train_u6lujuX_CVtuZ9i (1).csv")
except FileNotFoundError:
    print("[INFO] train_u6lujuX_CVtuZ9i (1).csv not found — generating synthetic stand-in (614 rows).")
    rng2 = np.random.default_rng(1)
    n = 614
    loan = pd.DataFrame({
        "Gender":            rng2.choice(["Male","Female"], n),
        "Married":           rng2.choice(["Yes","No"], n),
        "Dependents":        rng2.choice(["0","1","2","3+"], n),
        "Education":         rng2.choice(["Graduate","Not Graduate"], n),
        "Self_Employed":     rng2.choice(["Yes","No"], n),
        "ApplicantIncome":   rng2.integers(1000, 15000, n),
        "CoapplicantIncome": rng2.integers(0, 8000, n),
        "LoanAmount":        rng2.integers(50, 700, n),
        "Loan_Amount_Term":  rng2.choice([120,180,240,300,360,480], n),
        "Credit_History":    rng2.choice([0.0, 1.0], n),
        "Property_Area":     rng2.choice(["Urban","Semiurban","Rural"], n),
        "Loan_Status":       rng2.choice(["Y","N"], n, p=[0.69, 0.31]),
    })

print(f"\nDataset shape : {loan.shape}")
print(f"Target counts :\n{loan['Loan_Status'].value_counts().to_string()}\n")

# Drop Loan_ID if present
if "Loan_ID" in loan.columns:
    loan = loan.drop(columns=["Loan_ID"])

loan = encode_categorical(loan)
loan = impute_df(loan)

X_loan = loan.drop(columns=["Loan_Status"])
y_loan = loan["Loan_Status"]

X_tr_l, X_te_l, y_tr_l, y_te_l = train_test_split(
    X_loan, y_loan, test_size=0.20, random_state=42, stratify=y_loan
)

# ---- Best single Decision Tree (tuned) ----
dt_loan = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
dt_loan.fit(X_tr_l, y_tr_l)
dt_train_err = 1 - accuracy_score(y_tr_l, dt_loan.predict(X_tr_l))
dt_test_err  = 1 - accuracy_score(y_te_l, dt_loan.predict(X_te_l))
print(f"Best DT (max_depth=5)  — train err: {dt_train_err:.4f}  |  test err: {dt_test_err:.4f}")

# ---- Random Forest sweep over n_estimators ----
n_trees_range = [10, 25, 50, 100, 150, 200, 300]
rf_train_errs, rf_test_errs = [], []

for n_trees in n_trees_range:
    rf = RandomForestClassifier(n_estimators=n_trees, max_features="sqrt",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_tr_l, y_tr_l)
    rf_train_errs.append(1 - accuracy_score(y_tr_l, rf.predict(X_tr_l)))
    rf_test_errs.append (1 - accuracy_score(y_te_l, rf.predict(X_te_l)))
    print(f"  RF n_trees={n_trees:>3d}  |  train err={rf_train_errs[-1]:.4f}  |  test err={rf_test_errs[-1]:.4f}")

# ---- Best RF ----
best_rf_idx   = np.argmin(rf_test_errs)
best_rf_trees = n_trees_range[best_rf_idx]
best_rf_err   = rf_test_errs[best_rf_idx]

rf_best = RandomForestClassifier(n_estimators=best_rf_trees, max_features="sqrt",
                                  random_state=42, n_jobs=-1)
rf_best.fit(X_tr_l, y_tr_l)
print(f"\nBest RF: n_estimators={best_rf_trees}, test error={best_rf_err:.4f}")
print(f"Improvement over best DT: {(dt_test_err - best_rf_err)*100:.2f} pp")
print(f"\nClassification Report (RF, test set):\n{classification_report(y_te_l, rf_best.predict(X_te_l))}")

# ---- Feature importances ----
feat_imp   = pd.Series(rf_best.feature_importances_, index=X_loan.columns).sort_values(ascending=False)

# ---- Plot part (d) ----
fig_d = plt.figure(figsize=(14, 5))
gs    = gridspec.GridSpec(1, 2, figure=fig_d)
fig_d.suptitle("Part (d) — Random Forest | Loan Prediction Dataset", fontsize=13)

ax_rf = fig_d.add_subplot(gs[0])
ax_rf.plot(n_trees_range, rf_train_errs, "o-", color="steelblue", label="RF train error")
ax_rf.plot(n_trees_range, rf_test_errs,  "s--", color="tomato",   label="RF test error")
ax_rf.axhline(dt_test_err, linestyle=":", color="darkorange", label=f"Best DT test err ({dt_test_err:.3f})")
ax_rf.set_xlabel("Number of trees")
ax_rf.set_ylabel("Error rate")
ax_rf.set_title("RF error vs. number of trees")
ax_rf.legend()
ax_rf.grid(alpha=0.3)

ax_fi = fig_d.add_subplot(gs[1])
ax_fi.barh(feat_imp.index[:10][::-1], feat_imp.values[:10][::-1], color="steelblue")
ax_fi.set_xlabel("Feature importance (Gini)")
ax_fi.set_title("Top-10 feature importances")
ax_fi.grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("part_d_random_forest.png", dpi=150)
plt.show()
print("[Saved] part_d_random_forest.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY — All Results")
print("=" * 65)
summary = pd.DataFrame({
    "Model":       ["DT Baseline (Mush)", "DT Pruned depth=5 (Loan)",
                    f"Best k-NN k={best_k} (Mush)", f"Best RF n={best_rf_trees} (Loan)"],
    "Train Error": [f"{train_err_base:.4f}", f"{dt_train_err:.4f}",
                    "—", f"{rf_train_errs[best_rf_idx]:.4f}"],
    "Test Error":  [f"{test_err_base:.4f}", f"{dt_test_err:.4f}",
                    f"{best_knn_err:.4f}", f"{best_rf_err:.4f}"],
})
print(summary.to_string(index=False))
print("\n[Done] All figures saved.")