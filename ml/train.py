"""
Train the deep learning models behind the "Model" page of Voltwise.

Same dataset and approach as DL Experiments 2-7 (Devansh Pant, 16010423028, B-1):
  FFNN (Exp 2) -> DNN (Exp 3) -> regularised DNN + 1D CNN (Exp 5) -> LSTM (Exp 7),
plus the Low / Medium / High efficiency classifier from Exp 2.

Run from the project root:
    ml/.venv/Scripts/python ml/train.py
It writes the graphs and metrics.json into app/static/ml/, which the web page reads.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import callbacks, layers, models, regularizers

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ev_efficiency.csv"
OUT = ROOT / "app" / "static" / "ml"
SAVE = ROOT / "ml" / "models"
OUT.mkdir(parents=True, exist_ok=True)
SAVE.mkdir(parents=True, exist_ok=True)

SEED = 7
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

TARGET = "Energy Efficiency (km/kWh)"
ROSE, ROSE_LIGHT, PLUM, GREY = "#d6487a", "#eea6bf", "#6b2f4a", "#9b8a92"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.dpi": 110})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ 1. data
df = pd.read_csv(DATA)
df["Make"] = df["Make"].str.strip()            # "Tesla " and "Ford " appear with trailing spaces
print("Dataset:", df.shape)

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(df[TARGET], bins=35, color=ROSE, edgecolor="white")
ax[0].axvline(df[TARGET].mean(), color=PLUM, ls="--", label=f"mean {df[TARGET].mean():.2f}")
ax[0].set(title="Distribution of energy efficiency", xlabel="km/kWh", ylabel="vehicles")
ax[0].legend()
yearly = df.groupby("Model year")[TARGET].mean()
ax[1].plot(yearly.index, yearly.values, marker="o", color=ROSE)
ax[1].set(title="Average efficiency by model year", xlabel="Model year", ylabel="km/kWh")
ax[1].grid(alpha=.3)
save(fig, "01_dataset.png")

num_cols = ["Model year", "Motor (kW)", "Recharge time (h)", TARGET]
fig, ax = plt.subplots(figsize=(5.6, 4.4))
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap=sns.light_palette(ROSE, as_cmap=True),
            ax=ax, cbar=False, square=True)
ax.set_title("Correlation between numeric columns")
save(fig, "02_correlation.png")


# ------------------------------------------------------------------ 2. encoding (as in Exp 3)
def encode(frame):
    num = pd.DataFrame({
        "Model year": frame["Model year"],
        "Motor (kW)": frame["Motor (kW)"],
        "Recharge time (h)": frame["Recharge time (h)"],
        "Power rate": frame["Motor (kW)"] / frame["Recharge time (h)"],
    })
    cls = pd.get_dummies(frame["Vehicle class"], prefix="cls")
    make = pd.get_dummies(frame["Make"], prefix="make")
    return pd.concat([num, cls, make], axis=1).astype("float32")


X = encode(df)
y = df[TARGET].values.astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=SEED)
# scale only the 4 numeric columns; the one-hot columns stay 0/1
# (standardising rare one-hot makes turns them into huge values that destabilise dropout)
scaler = StandardScaler().fit(X_train[:, :4])
X_train[:, :4], X_test[:, :4] = scaler.transform(X_train[:, :4]), scaler.transform(X_test[:, :4])
N_FEATURES = X_train.shape[1]
print("Input shape:", (N_FEATURES,), "train", len(X_train), "test", len(X_test))


# ------------------------------------------------------------------ 3. regression models
def ffnn():
    return models.Sequential([layers.Input((N_FEATURES,)),
                              layers.Dense(16, activation="tanh", name="hidden"),
                              layers.Dense(1, name="output")], name="FFNN")


def dnn():
    return models.Sequential([layers.Input((N_FEATURES,)),
                              layers.Dense(64, activation="relu"),
                              layers.Dense(32, activation="relu"),
                              layers.Dense(16, activation="relu"),
                              layers.Dense(1)], name="DNN")


def dnn_reg():
    l2 = regularizers.l2(1e-4)
    return models.Sequential([layers.Input((N_FEATURES,)),
                              layers.Dense(64, activation="relu", kernel_regularizer=l2),
                              layers.Dropout(0.2),
                              layers.Dense(32, activation="relu", kernel_regularizer=l2),
                              layers.Dropout(0.2),
                              layers.Dense(16, activation="relu"),
                              layers.Dense(1)], name="DNN_Dropout")


def cnn():
    return models.Sequential([layers.Input((N_FEATURES, 1)),
                              layers.Conv1D(32, 3, activation="relu", padding="same"),
                              layers.MaxPooling1D(2),
                              layers.Conv1D(16, 3, activation="relu", padding="same"),
                              layers.MaxPooling1D(2),
                              layers.Flatten(),
                              layers.Dense(32, activation="relu"),
                              layers.Dense(1)], name="CNN_1D")


SPECS = [
    ("FFNN", "Exp 2", "Dense(16, tanh) → Dense(1)", ffnn, False),
    ("DNN", "Exp 3", "Dense(64) → Dense(32) → Dense(16), ReLU → Dense(1)", dnn, False),
    ("DNN + Dropout", "Exp 5", "DNN with Dropout(0.2) and L2(1e-4)", dnn_reg, False),
    ("1D CNN", "Exp 5", "Conv1D(32) → Pool → Conv1D(16) → Pool → Dense(32) → Dense(1)", cnn, True),
]
EPOCHS, BATCH, LR, PATIENCE = 300, 32, 1e-3, 25


def scores(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {"mae": round(float(mean_absolute_error(y_true, pred)), 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(y_true, pred)), 4)}


results, histories, preds = [], {}, {}

lin = LinearRegression().fit(X_train, y_train)
results.append({"name": "Linear regression", "exp": "Baseline", "arch": "scikit-learn LinearRegression",
                "params": int(N_FEATURES + 1), "epochs": "-", "seconds": 0.0,
                "train": scores(y_train, lin.predict(X_train)), "test": scores(y_test, lin.predict(X_test))})

for name, exp, arch, build, as_seq in SPECS:
    tf.keras.utils.set_random_seed(SEED)
    m = build()
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse", metrics=["mae"])
    xtr = X_train[..., None] if as_seq else X_train
    xte = X_test[..., None] if as_seq else X_test
    t0 = time.time()
    h = m.fit(xtr, y_train, validation_data=(xte, y_test), epochs=EPOCHS, batch_size=BATCH, verbose=0,
              callbacks=[callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    secs = time.time() - t0
    p_tr, p_te = m.predict(xtr, verbose=0).ravel(), m.predict(xte, verbose=0).ravel()
    histories[name], preds[name] = h.history, p_te
    r = {"name": name, "exp": exp, "arch": arch, "params": int(m.count_params()),
         "epochs": len(h.history["loss"]), "seconds": round(secs, 1),
         "train": scores(y_train, p_tr), "test": scores(y_test, p_te)}
    results.append(r)
    m.save(SAVE / f"{m.name}.keras")
    print(f"{name:14s} params {r['params']:6d} epochs {r['epochs']:3d}  test {r['test']}")

best = max((r for r in results if r["exp"] != "Baseline"), key=lambda r: r["test"]["r2"])
print("Best model:", best["name"])

# learning curves
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for a, (name, *_ ) in zip(axes.ravel(), SPECS):
    hist = histories[name]
    a.plot(hist["loss"], color=ROSE, label="training")
    a.plot(hist["val_loss"], color=PLUM, ls="--", label="validation")
    a.set(title=f"{name}: loss (MSE)", xlabel="epoch", ylabel="MSE", ylim=(0, max(1.0, min(hist["val_loss"]) * 4)))
    a.grid(alpha=.3)
    a.legend()
save(fig, "03_learning_curves.png")

# comparison
names = [r["name"] for r in results]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
colors = [ROSE if n == best["name"] else ROSE_LIGHT for n in names]
ax[0].barh(names, [r["test"]["mae"] for r in results], color=colors)
ax[0].set(title="Test MAE (lower is better)", xlabel="km/kWh")
ax[0].invert_yaxis()
ax[1].barh(names, [r["test"]["r2"] for r in results], color=colors)
ax[1].set(title="Test R² (higher is better)", xlim=(0, 1))
ax[1].invert_yaxis()
for a, key in ((ax[0], "mae"), (ax[1], "r2")):
    for i, r in enumerate(results):
        a.text(r["test"][key], i, f" {r['test'][key]:.3f}", va="center", fontsize=9)
save(fig, "04_model_comparison.png")

# best model diagnostics
bp = preds[best["name"]]
fig, ax = plt.subplots(figsize=(5.4, 5))
ax.scatter(y_test, bp, s=18, alpha=.65, color=ROSE)
lim = [y_test.min() - .2, y_test.max() + .2]
ax.plot(lim, lim, "--", color=PLUM, label="perfect prediction")
ax.set(title=f"{best['name']}: predicted vs actual (R² = {best['test']['r2']:.3f})",
       xlabel="actual km/kWh", ylabel="predicted km/kWh", xlim=lim, ylim=lim)
ax.legend()
ax.grid(alpha=.3)
save(fig, "05_pred_vs_actual.png")

res = bp - y_test
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(res, bins=30, color=ROSE, edgecolor="white")
ax[0].axvline(0, color=PLUM, ls="--")
ax[0].set(title="Distribution of residuals", xlabel="predicted − actual (km/kWh)")
ax[1].scatter(bp, res, s=16, alpha=.6, color=ROSE)
ax[1].axhline(0, color=PLUM, ls="--")
ax[1].set(title="Residuals vs predicted", xlabel="predicted km/kWh", ylabel="residual")
ax[1].grid(alpha=.3)
save(fig, "06_residuals.png")


# ------------------------------------------------------------------ 4. LSTM time series (as in Exp 7)
ts = df.sort_values("Model year", kind="stable").reset_index(drop=True)
series = ts[TARGET].astype("float32").values.reshape(-1, 1)
mm = MinMaxScaler().fit(series)
scaled = mm.transform(series)
LOOK_BACK = 12
Xs = np.array([scaled[i:i + LOOK_BACK, 0] for i in range(len(scaled) - LOOK_BACK)])[..., None]
ys = np.array([scaled[i + LOOK_BACK, 0] for i in range(len(scaled) - LOOK_BACK)])
split = int(len(Xs) * 0.8)
tf.keras.utils.set_random_seed(SEED)
lstm = models.Sequential([layers.Input((LOOK_BACK, 1)), layers.LSTM(50, activation="tanh"),
                          layers.Dense(25, activation="relu"), layers.Dense(1)], name="LSTM")
lstm.compile(optimizer="adam", loss="mse", metrics=["mae"])
t0 = time.time()
lh = lstm.fit(Xs[:split], ys[:split], validation_data=(Xs[split:], ys[split:]), epochs=60,
              batch_size=32, verbose=0)
lstm_secs = time.time() - t0
inv = lambda a: mm.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()
tr_p, te_p = inv(lstm.predict(Xs[:split], verbose=0)), inv(lstm.predict(Xs[split:], verbose=0))
tr_y, te_y = inv(ys[:split]), inv(ys[split:])
lstm_res = {"look_back": LOOK_BACK, "params": int(lstm.count_params()), "epochs": 60,
            "seconds": round(lstm_secs, 1), "train": scores(tr_y, tr_p), "test": scores(te_y, te_p),
            "train_size": int(split), "test_size": int(len(Xs) - split)}
lstm.save(SAVE / "LSTM.keras")
print("LSTM", lstm_res["test"])

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})
idx = np.arange(LOOK_BACK, LOOK_BACK + len(Xs))
ax[0].plot(idx, inv(ys), color="#cfc3c8", lw=1, label="actual")
ax[0].plot(idx[:split], tr_p, color=ROSE, lw=1.2, label="train prediction")
ax[0].plot(idx[split:], te_p, color=PLUM, lw=1.2, label="test prediction")
ax[0].axvline(idx[split], color=GREY, ls=":")
ax[0].set(title="LSTM predictions (rows ordered by model year)", xlabel="record index", ylabel="km/kWh")
ax[0].legend(loc="lower left", fontsize=8)
ax[1].plot(lh.history["loss"], color=ROSE, label="training")
ax[1].plot(lh.history["val_loss"], color=PLUM, ls="--", label="validation")
ax[1].set(title="LSTM loss (scaled MSE)", xlabel="epoch")
ax[1].legend()
ax[1].grid(alpha=.3)
save(fig, "07_lstm.png")


# ------------------------------------------------------------------ 5. classification (as in Exp 2)
LABELS = ["Low", "Medium", "High"]
to_class = lambda v: np.where(v < 4.8, 0, np.where(v < 5.3, 1, 2))
yc_train, yc_test = to_class(y_train), to_class(y_test)
tf.keras.utils.set_random_seed(SEED)
clf = models.Sequential([layers.Input((N_FEATURES,)), layers.Dense(64, activation="relu"),
                         layers.Dropout(0.2), layers.Dense(32, activation="relu"),
                         layers.Dense(3, activation="softmax")], name="Classifier")
clf.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="sparse_categorical_crossentropy",
            metrics=["accuracy"])
ch = clf.fit(X_train, yc_train, validation_data=(X_test, yc_test), epochs=EPOCHS, batch_size=BATCH,
             verbose=0, callbacks=[callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
yc_pred = clf.predict(X_test, verbose=0).argmax(axis=1)
report = classification_report(yc_test, yc_pred, target_names=LABELS, output_dict=True, zero_division=0)
clf_res = {"accuracy": round(float(accuracy_score(yc_test, yc_pred)), 4),
           "train_accuracy": round(float(ch.history["accuracy"][-1]), 4),
           "epochs": len(ch.history["loss"]), "params": int(clf.count_params()),
           "per_class": {k: {"precision": round(report[k]["precision"], 3), "recall": round(report[k]["recall"], 3),
                             "f1": round(report[k]["f1-score"], 3), "support": int(report[k]["support"])}
                         for k in LABELS},
           "class_counts": {LABELS[i]: int((to_class(y) == i).sum()) for i in range(3)}}
clf.save(SAVE / "Classifier.keras")
print("Classifier accuracy", clf_res["accuracy"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cm = confusion_matrix(yc_test, yc_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(ROSE, as_cmap=True), cbar=False,
            xticklabels=LABELS, yticklabels=LABELS, ax=ax[0])
ax[0].set(title=f"Confusion matrix (accuracy {clf_res['accuracy']:.1%})", xlabel="predicted", ylabel="actual")
ax[1].plot(ch.history["accuracy"], color=ROSE, label="training")
ax[1].plot(ch.history["val_accuracy"], color=PLUM, ls="--", label="validation")
ax[1].set(title="Classifier accuracy per epoch", xlabel="epoch", ylim=(0, 1))
ax[1].legend()
ax[1].grid(alpha=.3)
save(fig, "08_classification.png")


# ------------------------------------------------------------------ 6. write metrics for the web page
metrics = {
    "trained_on": date.today().isoformat(),
    "dataset": {"rows": int(len(df)), "makes": int(df["Make"].nunique()),
                "classes": int(df["Vehicle class"].nunique()),
                "years": [int(df["Model year"].min()), int(df["Model year"].max())],
                "target_mean": round(float(y.mean()), 3), "features": int(N_FEATURES),
                "train": int(len(X_train)), "test": int(len(X_test))},
    "hyperparameters": {"optimizer": "Adam", "learning_rate": LR, "loss": "MSE", "batch_size": BATCH,
                        "max_epochs": EPOCHS, "early_stopping_patience": PATIENCE, "test_split": 0.2,
                        "seed": SEED},
    "regression": results,
    "best": best["name"],
    "lstm": lstm_res,
    "classification": clf_res,
    "versions": {"python": "3.12", "tensorflow": tf.__version__, "keras": tf.keras.__version__,
                 "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
                 "matplotlib": matplotlib.__version__, "seaborn": sns.__version__},
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved graphs and metrics to", OUT)
