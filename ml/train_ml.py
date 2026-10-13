"""
Classical machine learning on the EV dataset, following ML Experiments 1-8
(Devansh Pant, 16010423028, B-1). scikit-learn only, no TensorFlow needed.

Run from the project root (after ml/train.py, so the DL numbers exist for the comparison):
    ml/.venv/Scripts/python ml/train_ml.py
Writes ml_*.png and ml_metrics.json into app/static/ml/.
"""
import json
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ev_efficiency.csv"
OUT = ROOT / "app" / "static" / "ml"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7
TARGET = "Energy Efficiency (km/kWh)"
ROSE, ROSE_LIGHT, PLUM, GREY = "#d6487a", "#eea6bf", "#6b2f4a", "#9b8a92"
PALETTE = [ROSE, PLUM, "#e9a03b", "#5b8fb9", "#7aa874"]
LABELS = ["Low", "Medium", "High"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})
pink_map = sns.light_palette(ROSE, as_cmap=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def r(v, n=4):
    return round(float(v), n)


timings = {}
t_all = time.time()

# =============================================================== Exp 1: preprocessing
df = pd.read_csv(DATA)
missing = int(df.isnull().sum().sum())
df["Make"] = df["Make"].str.strip()
df["Power rate"] = df["Motor (kW)"] / df["Recharge time (h)"]

eq_width = pd.cut(df[TARGET], bins=4, labels=["Low", "Medium", "High", "Very high"])
eq_freq = pd.qcut(df[TARGET], q=4, labels=["Low", "Medium", "High", "Very high"])
motor = df[["Motor (kW)"]].values
mm = MinMaxScaler().fit_transform(motor).ravel()
zs = StandardScaler().fit_transform(motor).ravel()

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].hist(motor.ravel(), bins=30, color=GREY, edgecolor="white")
ax[0].set(title="Motor (kW) — raw", xlabel="kW")
ax[1].hist(mm, bins=30, color=ROSE_LIGHT, edgecolor="white", label="Min-Max (0 to 1)")
ax[1].hist(zs, bins=30, color=ROSE, edgecolor="white", alpha=.75, label="Z-score (mean 0, sd 1)")
ax[1].set(title="After normalisation", xlabel="scaled value")
ax[1].legend(fontsize=8)
x = np.arange(4)
ax[2].bar(x - .2, eq_width.value_counts().sort_index().values, .4, color=ROSE, label="equal width")
ax[2].bar(x + .2, eq_freq.value_counts().sort_index().values, .4, color=PLUM, label="equal frequency")
ax[2].set_xticks(x, ["Low", "Medium", "High", "Very high"])
ax[2].set(title="Discretising efficiency into 4 bins", ylabel="vehicles")
ax[2].legend(fontsize=8)
save(fig, "ml_01_preprocessing.png")

pre = {"missing_values": missing, "makes_after_cleaning": int(df["Make"].nunique()),
       "equal_width": {k: int(v) for k, v in eq_width.value_counts().sort_index().items()},
       "equal_frequency": {k: int(v) for k, v in eq_freq.value_counts().sort_index().items()},
       "motor_raw_range": [int(motor.min()), int(motor.max())]}

# shared encoded matrix (same as the DL page) and split
NUM = ["Model year", "Motor (kW)", "Recharge time (h)", "Power rate"]
X_all = pd.concat([df[NUM], pd.get_dummies(df["Vehicle class"], prefix="cls"),
                   pd.get_dummies(df["Make"], prefix="make")], axis=1).astype(float)
y_all = df[TARGET].values
pre["features"] = int(X_all.shape[1])
X_train, X_test, y_train, y_test = train_test_split(X_all.values, y_all, test_size=0.2, random_state=SEED)
sc = StandardScaler().fit(X_train[:, :4])
X_train[:, :4], X_test[:, :4] = sc.transform(X_train[:, :4]), sc.transform(X_test[:, :4])
to_class = lambda v: np.where(v < 4.8, 0, np.where(v < 5.3, 1, 2))
yc_train, yc_test = to_class(y_train), to_class(y_test)

# =============================================================== Exp 2 + 3: Naive Bayes, Decision Tree
t0 = time.time()
# Gaussian NB assumes each feature is normally distributed, so it gets the 4 numeric columns
# (the 0/1 one-hot columns break that assumption; with them accuracy drops to ~60%)
nb = GaussianNB().fit(X_train[:, :4], yc_train)
nb_pred = nb.predict(X_test[:, :4])
nb_all_acc = accuracy_score(yc_test, GaussianNB().fit(X_train, yc_train).predict(X_test))

# depth is chosen with 5-fold cross-validation on the training set only, never on the test set
depths = list(range(1, 13))
dt_train_acc, dt_cv_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=SEED)
    dt_cv_acc.append(cross_val_score(m, X_train, yc_train, cv=5).mean())
    dt_train_acc.append(accuracy_score(yc_train, m.fit(X_train, yc_train).predict(X_train)))
best_depth = depths[int(np.argmax(dt_cv_acc))]
dt = DecisionTreeClassifier(max_depth=best_depth, criterion="gini", random_state=SEED).fit(X_train, yc_train)
dt_pred = dt.predict(X_test)
timings["classifiers"] = time.time() - t0


def cls_summary(pred):
    rep = classification_report(yc_test, pred, target_names=LABELS, output_dict=True, zero_division=0)
    return {"accuracy": r(accuracy_score(yc_test, pred)),
            "per_class": {k: {"precision": r(rep[k]["precision"], 3), "recall": r(rep[k]["recall"], 3),
                              "f1": r(rep[k]["f1-score"], 3)} for k in LABELS}}


fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, pred, title in ((ax[0], nb_pred, "Gaussian Naive Bayes"), (ax[1], dt_pred, f"Decision Tree (depth {best_depth})")):
    sns.heatmap(confusion_matrix(yc_test, pred), annot=True, fmt="d", cmap=pink_map, cbar=False,
                xticklabels=LABELS, yticklabels=LABELS, ax=a)
    a.set(title=f"{title}: {accuracy_score(yc_test, pred):.1%}", xlabel="predicted", ylabel="actual")
ax[2].plot(depths, dt_train_acc, marker="o", color=ROSE, label="training")
ax[2].plot(depths, dt_cv_acc, marker="o", color=PLUM, ls="--", label="5-fold cross-validation")
ax[2].axvline(best_depth, color=GREY, ls=":")
ax[2].set(title="Decision Tree: accuracy vs depth", xlabel="max_depth", ylabel="accuracy")
ax[2].legend()
ax[2].grid(alpha=.3)
save(fig, "ml_02_classifiers.png")

small_tree = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(X_train, yc_train)
fig, ax = plt.subplots(figsize=(15, 6))
plot_tree(small_tree, feature_names=list(X_all.columns), class_names=LABELS, filled=True, rounded=True,
          impurity=True, fontsize=8, ax=ax)
ax.set_title("Decision Tree (first 3 levels, Gini) — numeric columns are standardised")
save(fig, "ml_03_tree.png")

importances = sorted(zip(X_all.columns, dt.feature_importances_), key=lambda t: -t[1])[:5]

# =============================================================== Exp 4: gradient descent with / without scaling
gd_cols = ["Model year", "Motor (kW)", "Recharge time (h)"]
Xg = df[gd_cols].values.astype(float)
yg = df[TARGET].values


def gradient_descent(X, y, lr, iters):
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    costs = []
    for _ in range(iters):
        err = Xb @ w - y
        cost = (err @ err) / (2 * len(y))
        if not np.isfinite(cost) or cost > 1e12:
            costs.append(np.nan)
            break
        costs.append(cost)
        w -= lr * (Xb.T @ err) / len(y)
    return np.array(costs), w


ITERS = 1000
Xg_scaled = StandardScaler().fit_transform(Xg)
cost_scaled, w_scaled = gradient_descent(Xg_scaled, yg, 0.1, ITERS)
cost_raw_small, _ = gradient_descent(Xg, yg, 1e-8, ITERS)          # largest rate that does not blow up
cost_raw_big, _ = gradient_descent(Xg, yg, 1e-6, ITERS)            # diverges
closed = LinearRegression().fit(Xg_scaled, yg)
best_cost = mean_squared_error(yg, closed.predict(Xg_scaled)) / 2
conv_iter = int(np.argmax(cost_scaled <= best_cost * 1.001)) + 1

fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
ax[0].plot(cost_scaled, color=ROSE, lw=2, label="scaled features, lr = 0.1")
ax[0].plot(cost_raw_small, color=PLUM, lw=2, ls="--", label="raw features, lr = 1e-8")
ax[0].axhline(best_cost, color=GREY, ls=":", label="optimum (normal equation)")
ax[0].set(title="Cost J(w) per iteration", xlabel="iteration", ylabel="cost (MSE / 2)", yscale="log")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3)
ax[1].plot(cost_raw_big[:len(cost_raw_big)], color="#c2453b", lw=2)
ax[1].set(title="Raw features with lr = 1e-6: diverges", xlabel="iteration", ylabel="cost", yscale="log")
ax[1].grid(alpha=.3)
save(fig, "ml_04_gradient_descent.png")

gd = {"iterations": ITERS, "optimal_cost": r(best_cost), "scaled_final_cost": r(cost_scaled[-1]),
      "scaled_converged_at": conv_iter, "raw_final_cost": r(np.nanmin(cost_raw_small)),
      "raw_diverged_after": int(len(cost_raw_big)),
      "raw_ranges": {c: [r(df[c].min(), 1), r(df[c].max(), 1)] for c in gd_cols}}

# =============================================================== Exp 5: K-Means
km_cols = ["Motor (kW)", "Recharge time (h)", TARGET]
Xk = StandardScaler().fit_transform(df[km_cols])
wcss = [KMeans(n_clusters=k, random_state=0, n_init=10).fit(Xk).inertia_ for k in range(1, 11)]
sil = {k: r(silhouette_score(Xk, KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(Xk)), 3)
       for k in range(2, 8)}
K = 3
km = KMeans(n_clusters=K, random_state=0, n_init=10).fit(Xk)
df["cluster"] = km.labels_
prof = df.groupby("cluster")[km_cols].mean()
order = prof[TARGET].sort_values(ascending=False).index          # most efficient first
names = {order[0]: "Efficient, low power", order[1]: "Mid power", order[2]: "High power"}
clusters = [{"name": names[c], "vehicles": int((df["cluster"] == c).sum()),
             "motor": r(prof.loc[c, "Motor (kW)"], 1), "recharge": r(prof.loc[c, "Recharge time (h)"], 2),
             "efficiency": r(prof.loc[c, TARGET], 2)} for c in order]

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(range(1, 11), wcss, marker="o", color=ROSE)
ax[0].axvline(K, color=GREY, ls=":")
ax[0].set(title="Elbow method", xlabel="number of clusters K", ylabel="WCSS (inertia)")
ax[0].grid(alpha=.3)
for i, c in enumerate(order):
    s = df[df["cluster"] == c]
    ax[1].scatter(s["Motor (kW)"], s[TARGET], s=12, alpha=.65, color=PALETTE[i], label=names[c])
ax[1].set(title=f"K-Means clusters (K = {K}, silhouette {sil[K]:.2f})", xlabel="Motor (kW)", ylabel="km/kWh")
ax[1].legend(fontsize=8)
save(fig, "ml_05_kmeans.png")

# =============================================================== Exp 6: Gaussian Mixture Model
gm_cols = ["Model year", "Motor (kW)", "Recharge time (h)", TARGET]
Xm = StandardScaler().fit_transform(df[gm_cols])
bic = {n: r(GaussianMixture(n, covariance_type="full", random_state=42, n_init=5).fit(Xm).bic(Xm), 1)
       for n in range(1, 9)}
gmm = GaussianMixture(4, covariance_type="full", random_state=42, n_init=10).fit(Xm)
states = gmm.predict(Xm)
probs = gmm.predict_proba(Xm)
df["state"] = states
sprof = df.groupby("state")[gm_cols].mean()
pca2 = PCA(2, random_state=42).fit_transform(Xm)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for s in range(4):
    msk = states == s
    ax[0].scatter(pca2[msk, 0], pca2[msk, 1], s=12, alpha=.65, color=PALETTE[s], label=f"State {s}")
ax[0].set(title="GMM latent states (shown on 2 principal components)", xlabel="PC 1", ylabel="PC 2")
ax[0].legend(fontsize=8)
ax[1].plot(list(bic), list(bic.values()), marker="o", color=ROSE)
ax[1].axvline(4, color=GREY, ls=":")
ax[1].set(title="BIC for 1-8 components (lower is better)", xlabel="components", ylabel="BIC")
ax[1].grid(alpha=.3)
save(fig, "ml_06_gmm.png")

gm = {"components": 4, "converged": bool(gmm.converged_), "iterations": int(gmm.n_iter_),
      "aic": r(gmm.aic(Xm), 1), "bic": r(gmm.bic(Xm), 1), "silhouette": r(silhouette_score(Xm, states), 3),
      "avg_max_probability": r(probs.max(axis=1).mean(), 3),
      "states": [{"state": int(s), "vehicles": int((states == s).sum()), "year": r(sprof.loc[s, "Model year"], 1),
                  "motor": r(sprof.loc[s, "Motor (kW)"], 1), "recharge": r(sprof.loc[s, "Recharge time (h)"], 2),
                  "efficiency": r(sprof.loc[s, TARGET], 2)} for s in range(4)]}

# =============================================================== Exp 7: PCA vs t-SNE
Xd = X_all.values.copy()
Xd[:, :4] = StandardScaler().fit_transform(Xd[:, :4])
pca_full = PCA(random_state=SEED).fit(Xd)
cum = np.cumsum(pca_full.explained_variance_ratio_)
n90 = int(np.argmax(cum >= 0.90)) + 1
p2 = PCA(2, random_state=SEED).fit_transform(Xd)
t0 = time.time()
ts = TSNE(2, perplexity=30, random_state=SEED, init="pca").fit_transform(Xd)
timings["tsne"] = time.time() - t0
cls_all = to_class(y_all)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].plot(range(1, len(cum) + 1), cum, color=ROSE, lw=2)
ax[0].axhline(.9, color=GREY, ls=":")
ax[0].axvline(n90, color=GREY, ls=":")
ax[0].set(title=f"PCA: {n90} components keep 90% variance", xlabel="components", ylabel="cumulative variance")
ax[0].grid(alpha=.3)
for a, emb, title in ((ax[1], p2, "PCA (linear) — 2 components"), (ax[2], ts, "t-SNE (non-linear) — 2 dimensions")):
    for i, lab in enumerate(LABELS):
        msk = cls_all == i
        a.scatter(emb[msk, 0], emb[msk, 1], s=9, alpha=.6, color=[GREY, ROSE_LIGHT, PLUM][i], label=lab)
    a.set(title=title)
    a.legend(fontsize=8, title="efficiency")
save(fig, "ml_07_dimred.png")

dr = {"original_dims": int(Xd.shape[1]), "components_90": n90,
      "pc1_pc2_variance": r(cum[1], 3), "tsne_seconds": r(timings["tsne"], 1)}

# =============================================================== Exp 8: SVM kernels
median_eff = float(np.median(y_all))
ys = (y_all >= median_eff).astype(int)
Xs = df[["Motor (kW)", "Recharge time (h)"]].values
Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(Xs, ys, test_size=0.2, random_state=42, stratify=ys)
ssc = StandardScaler().fit(Xs_tr)
Xs_tr, Xs_te = ssc.transform(Xs_tr), ssc.transform(Xs_te)
svm_res, svms = [], {}
for k in ["linear", "poly", "rbf"]:
    m = SVC(kernel=k, C=1.0, gamma="scale").fit(Xs_tr, ys_tr)
    svms[k] = m
    svm_res.append({"kernel": k, "accuracy": r(accuracy_score(ys_te, m.predict(Xs_te))),
                    "support_vectors": int(m.n_support_.sum())})
best_svm = max(svm_res, key=lambda d: d["accuracy"])

xx, yy = np.meshgrid(np.linspace(Xs_tr[:, 0].min() - .5, Xs_tr[:, 0].max() + .5, 300),
                     np.linspace(Xs_tr[:, 1].min() - .5, Xs_tr[:, 1].max() + .5, 300))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
cmap_bg = matplotlib.colors.ListedColormap(["#efe7ea", "#f9d3e0"])
for a, res in zip(ax, svm_res):
    Z = svms[res["kernel"]].predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    a.contourf(xx, yy, Z, cmap=cmap_bg)
    a.scatter(Xs_tr[ys_tr == 0, 0], Xs_tr[ys_tr == 0, 1], s=9, color=GREY, label="less efficient")
    a.scatter(Xs_tr[ys_tr == 1, 0], Xs_tr[ys_tr == 1, 1], s=9, color=ROSE, label="efficient")
    a.set(title=f"{res['kernel']} kernel: {res['accuracy']:.1%}", xlabel="motor (scaled)", ylabel="recharge (scaled)")
ax[0].legend(fontsize=8, loc="upper left")
save(fig, "ml_08_svm.png")

svm = {"threshold": r(median_eff, 2), "kernels": svm_res, "best": best_svm["kernel"]}

# =============================================================== classical regressors vs deep learning
t0 = time.time()
regs = [
    ("Linear regression", LinearRegression()),
    ("Decision Tree", DecisionTreeRegressor(max_depth=8, random_state=SEED)),
    ("Random Forest", RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)),
    ("Gradient Boosting", GradientBoostingRegressor(random_state=SEED)),
    ("KNN (k = 5)", KNeighborsRegressor(n_neighbors=5)),
    ("SVR (RBF)", SVR(C=3.0, epsilon=0.05)),
]
reg_res = []
for name, m in regs:
    m.fit(X_train, y_train)
    p = m.predict(X_test)
    reg_res.append({"name": name, "family": "ML", "mae": r(mean_absolute_error(y_test, p)),
                    "rmse": r(np.sqrt(mean_squared_error(y_test, p))), "r2": r(r2_score(y_test, p)),
                    "train_r2": r(r2_score(y_train, m.predict(X_train)))})
timings["regressors"] = time.time() - t0
best_ml = max(reg_res, key=lambda d: d["r2"])

dl_path = OUT / "metrics.json"
dl_rows = []
if dl_path.exists():
    dl = json.loads(dl_path.read_text(encoding="utf-8"))
    dl_rows = [{"name": x["name"], "family": "DL", "mae": x["test"]["mae"], "rmse": x["test"]["rmse"],
                "r2": x["test"]["r2"]} for x in dl["regression"] if x["exp"] != "Baseline"]
both = sorted(reg_res + dl_rows, key=lambda d: d["r2"])

fig, ax = plt.subplots(figsize=(10, 0.42 * len(both) + 1.2))
ax.barh([d["name"] for d in both], [d["r2"] for d in both],
        color=[ROSE if d["family"] == "ML" else PLUM for d in both])
for i, d in enumerate(both):
    ax.text(d["r2"], i, f" {d['r2']:.3f}", va="center", fontsize=9)
ax.set(title="Test R² on the same 240 cars — classical ML (pink) vs deep learning (plum)", xlim=(0, 1.05))
save(fig, "ml_09_ml_vs_dl.png")

rf = dict(regs)["Random Forest"]
imp = sorted(zip(X_all.columns, rf.feature_importances_), key=lambda t: -t[1])[:10]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh([n for n, _ in imp][::-1], [v for _, v in imp][::-1], color=ROSE)
ax.set(title="Random Forest: 10 most important features", xlabel="importance")
save(fig, "ml_10_feature_importance.png")

metrics = {
    "trained_on": date.today().isoformat(),
    "seconds_total": r(time.time() - t_all, 1),
    "split": {"train": int(len(X_train)), "test": int(len(X_test))},
    "preprocessing": pre,
    "naive_bayes": {**cls_summary(nb_pred), "all_features_accuracy": r(nb_all_acc)},
    "decision_tree": {**cls_summary(dt_pred), "depth": best_depth, "cv_accuracy": r(max(dt_cv_acc)),
                      "top_features": [{"name": n, "importance": r(v, 3)} for n, v in importances]},
    "gradient_descent": gd,
    "kmeans": {"k": K, "silhouette": sil, "clusters": clusters},
    "gmm": gm,
    "dimred": dr,
    "svm": svm,
    "regression": reg_res,
    "best_ml": best_ml["name"],
    "dl_rows": dl_rows,
    "feature_importance": [{"name": n, "importance": r(v, 3)} for n, v in imp],
    "versions": {"scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                 "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
}
(OUT / "ml_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"NB acc {metrics['naive_bayes']['accuracy']}  DT acc {metrics['decision_tree']['accuracy']} (depth {best_depth})")
print(f"GD scaled converged at {conv_iter}, raw diverged after {gd['raw_diverged_after']}")
print(f"KMeans silhouette K=3 {sil[3]}  GMM silhouette {gm['silhouette']}  PCA 90% at {n90}")
print("SVM", svm_res)
for d in reg_res:
    print(f"{d['name']:18s} MAE {d['mae']:.4f} R2 {d['r2']:.4f}")
print("Best ML:", best_ml["name"], "| total", metrics["seconds_total"], "s")
