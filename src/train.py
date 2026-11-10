"""
train.py
--------
Stage 5 + 6: train, tune, evaluate and save the PNN.

Run from the project root:
    python -m src.train

What it does (in order)
  1. Reads outputs/splits.csv (made by src.preprocessing) and extracts features
  2. Fits the scaler on TRAIN only
  3. Tunes sigma and the prior type on the VALIDATION set only
  4. Evaluates the chosen model ONCE on the TEST set
  5. Trains a second PNN that recognises the fruit type (Apple/Banana/Strawberry)
  6. Saves the model to models/ and every real result to outputs/

The test set is touched only in step 4, after all choices are frozen.
"""

import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")                     # draw plots to files, no pop-up window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from src.feature_extraction import extract_from_path, feature_block_slices
from src.pnn import PNN, BlockScaler
from src.preprocessing import (CLASS_NAMES, FRUITS, IMAGE_SIZE, OUTPUT_DIR,
                               PROJECT_ROOT, RANDOM_SEED, load_splits)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "pnn_model.joblib"
FEATURE_CACHE = OUTPUT_DIR / "features_cache.npz"

SIGMA_GRID = np.round(np.geomspace(0.03, 2.5, 30), 4)   # 30 values, log-spaced
PRIOR_OPTIONS = ["empirical", "uniform"]


# ----------------------------------------------------------------------------
# 1. Features (cached so re-running is quick)
# ----------------------------------------------------------------------------
def get_features(splits):
    paths = splits["path"].to_numpy()
    if FEATURE_CACHE.exists():
        cache = np.load(FEATURE_CACHE, allow_pickle=True)
        if np.array_equal(cache["paths"], paths):
            print("Loaded cached features.")
            return cache["X"]
    print("Extracting features from", len(paths), "images ...")
    X = np.stack([extract_from_path(p) for p in paths])
    OUTPUT_DIR.mkdir(exist_ok=True)
    np.savez(FEATURE_CACHE, X=X, paths=paths)
    return X


# ----------------------------------------------------------------------------
# 3. Tuning on validation data
# ----------------------------------------------------------------------------
def tune_pnn(X_tr, y_tr, X_va, y_va):
    """
    Try every (prior, sigma) pair. Score = macro-F1 on VALIDATION data
    (macro-F1 gives Fresh and Rotten equal weight, so it is not fooled by the
    imbalance). Ties are broken by the lower validation log-loss.
    """
    rows = []
    for prior in PRIOR_OPTIONS:
        for sigma in SIGMA_GRID:
            m = PNN(sigma=float(sigma), priors=prior).fit(X_tr, y_tr)
            proba = m.predict_proba(X_va)
            pred = m.classes_[np.argmax(proba, axis=1)]
            rows.append({
                "prior": prior, "sigma": float(sigma),
                "val_accuracy": accuracy_score(y_va, pred),
                "val_macro_f1": f1_score(y_va, pred, average="macro"),
                "val_log_loss": log_loss(y_va, np.clip(proba, 1e-12, 1),
                                         labels=m.classes_),
            })
    res = pd.DataFrame(rows)
    best = res.sort_values(["val_macro_f1", "val_log_loss"],
                           ascending=[False, True]).iloc[0]
    return res, best


# ----------------------------------------------------------------------------
# 4. Evaluation helpers
# ----------------------------------------------------------------------------
def plot_tuning(res, best, path, title):
    plt.figure(figsize=(7, 4.5))
    for prior, g in res.groupby("prior"):
        plt.plot(g["sigma"], g["val_macro_f1"], marker="o", ms=3, label=f"priors = {prior}")
    plt.axvline(best["sigma"], ls="--", c="grey", label=f"chosen sigma = {best['sigma']:.3g}")
    plt.xscale("log")
    plt.xlabel("sigma (log scale)")
    plt.ylabel("Validation macro-F1")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def evaluate_test(model, X_te, y_te, test_df):
    proba = model.predict_proba(X_te)
    pred = np.argmax(proba, axis=1)
    conf = proba.max(axis=1)

    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision_macro": precision_score(y_te, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, pred, average="macro", zero_division=0),
        "precision_rotten": precision_score(y_te, pred, pos_label=1, zero_division=0),
        "recall_rotten": recall_score(y_te, pred, pos_label=1, zero_division=0),
        "f1_rotten": f1_score(y_te, pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_te, proba[:, 1]),
        # confidence-vs-calibration diagnostics (see README)
        "mean_confidence": float(conf.mean()),
        "mean_confidence_when_correct": float(conf[pred == y_te].mean()),
        "mean_confidence_when_wrong": (float(conf[pred != y_te].mean())
                                       if (pred != y_te).any() else None),
        "brier_score": brier_score_loss(y_te, proba[:, 1]),
        "n_test": int(len(y_te)),
    }

    # confusion matrix plot
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("PNN confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # ROC curve plot (Rotten = positive class)
    fpr, tpr, _ = roc_curve(y_te, proba[:, 1])
    plt.figure(figsize=(4.8, 4.2))
    plt.plot(fpr, tpr, label=f"PNN (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", c="grey", label="chance")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve (positive class = Rotten)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    plt.close()

    report = classification_report(y_te, pred, target_names=CLASS_NAMES, digits=3)

    # per-fruit results
    rows = []
    for fruit in FRUITS:
        idx = (test_df["fruit"] == fruit).to_numpy()
        yt, yp = y_te[idx], pred[idx]
        rows.append({
            "fruit": fruit, "n_test": int(idx.sum()),
            "n_fresh": int((yt == 0).sum()), "n_rotten": int((yt == 1).sum()),
            "accuracy": accuracy_score(yt, yp),
            "rotten_precision": precision_score(yt, yp, pos_label=1, zero_division=0),
            "rotten_recall": recall_score(yt, yp, pos_label=1, zero_division=0),
            "rotten_f1": f1_score(yt, yp, pos_label=1, zero_division=0),
            "errors": int((yt != yp).sum()),
        })
    per_fruit = pd.DataFrame(rows)

    pred_table = test_df[["path", "fruit", "label"]].copy()
    pred_table["predicted"] = [CLASS_NAMES[i] for i in pred]
    pred_table["p_fresh"] = proba[:, 0]
    pred_table["p_rotten"] = proba[:, 1]
    return metrics, report, per_fruit, cm, pred_table


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    splits = load_splits()
    X = get_features(splits)
    tr = (splits["split"] == "train").to_numpy()
    va = (splits["split"] == "val").to_numpy()
    te = (splits["split"] == "test").to_numpy()
    y = splits["label_id"].to_numpy()
    print(f"train={tr.sum()}  val={va.sum()}  test={te.sum()}")

    # --- scaler: fitted on TRAIN only ---------------------------------------
    scaler = BlockScaler(feature_block_slices()).fit(X[tr])
    Xs = scaler.transform(X)

    # --- tune freshness PNN on VALIDATION only --------------------------------
    print("\nTuning sigma and priors on the validation set ...")
    res, best = tune_pnn(Xs[tr], y[tr], Xs[va], y[va])
    res.to_csv(OUTPUT_DIR / "tuning_results.csv", index=False)
    plot_tuning(res, best, OUTPUT_DIR / "sigma_tuning.png",
                "Freshness PNN: choosing sigma on validation data")
    print(f"Chosen: sigma={best['sigma']}  priors={best['prior']}  "
          f"(val macro-F1={best['val_macro_f1']:.3f})")

    # --- final freshness model (trained on TRAIN; test still untouched) --------
    model = PNN(sigma=float(best["sigma"]), priors=best["prior"]).fit(Xs[tr], y[tr])

    # --- one-time evaluation on TEST --------------------------------------------
    test_df = splits[te].reset_index(drop=True)
    metrics, report, per_fruit, cm, pred_table = evaluate_test(
        model, Xs[te], y[te], test_df)

    # --- second PNN: which fruit is it? (also a genuine PNN) ----------------------
    fruit_to_id = {f: i for i, f in enumerate(FRUITS)}
    yf = splits["fruit"].map(fruit_to_id).to_numpy()
    res_f, best_f = tune_pnn(Xs[tr], yf[tr], Xs[va], yf[va])
    res_f.to_csv(OUTPUT_DIR / "tuning_results_fruit_type.csv", index=False)
    fruit_model = PNN(sigma=float(best_f["sigma"]), priors=best_f["prior"]).fit(Xs[tr], yf[tr])
    fruit_pred = fruit_model.predict(Xs[te])
    metrics["fruit_type_accuracy_test"] = accuracy_score(yf[te], fruit_pred)

    # --- save everything ----------------------------------------------------------
    joblib.dump({
        "freshness_model": model,
        "fruit_model": fruit_model,
        "scaler": scaler,
        "class_names": CLASS_NAMES,
        "fruit_names": FRUITS,
        "image_size": IMAGE_SIZE,
        "info": {
            "freshness_sigma": float(best["sigma"]),
            "freshness_priors": best["prior"],
            "fruit_sigma": float(best_f["sigma"]),
            "fruit_priors": best_f["prior"],
            "n_train": int(tr.sum()), "n_features": int(X.shape[1]),
            "seed": RANDOM_SEED,
        },
    }, MODEL_PATH)

    metrics_out = {k: (None if v is None else float(v)) for k, v in metrics.items()}
    metrics_out["chosen_sigma"] = float(best["sigma"])
    metrics_out["chosen_priors"] = best["prior"]
    metrics_out["val_macro_f1_at_chosen"] = float(best["val_macro_f1"])
    metrics_out["confusion_matrix [[TN,FP],[FN,TP]] (Fresh=0, Rotten=1)"] = cm.tolist()
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    (OUTPUT_DIR / "classification_report.txt").write_text(report)
    per_fruit.to_csv(OUTPUT_DIR / "per_fruit_metrics.csv", index=False)
    pred_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # --- print a readable summary ---------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST-SET RESULTS  (unseen images, computed - not hard-coded)")
    print("=" * 60)
    print(report)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=[f"actual {c}" for c in CLASS_NAMES],
                       columns=[f"pred {c}" for c in CLASS_NAMES]))
    print(f"\nROC-AUC (Rotten positive): {metrics['roc_auc']:.3f}")
    print("\nPer-fruit results (small n - interpret with care):")
    print(per_fruit.round(3).to_string(index=False))
    print(f"\nFruit-type PNN test accuracy: {metrics['fruit_type_accuracy_test']:.3f}")
    print(f"\nMean confidence {metrics['mean_confidence']:.3f}  vs  accuracy "
          f"{metrics['accuracy']:.3f}   (Brier score {metrics['brier_score']:.3f})")
    print(f"\nModel saved to : {MODEL_PATH}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()