"""
make_report.py
--------------
Fills the "Results" section of README.md using the files that train.py actually
produced (outputs/metrics.json, per_fruit_metrics.csv, splits.csv ...).
Nothing is typed in by hand, so the README can never disagree with your run.

Run after training:
    python -m src.make_report
"""

import json

import pandas as pd

from src.preprocessing import CLASS_NAMES, OUTPUT_DIR, PROJECT_ROOT

README = PROJECT_ROOT / "README.md"
START, END = "<!-- RESULTS START -->", "<!-- RESULTS END -->"


def build_results_markdown():
    m = json.loads((OUTPUT_DIR / "metrics.json").read_text())
    per_fruit = pd.read_csv(OUTPUT_DIR / "per_fruit_metrics.csv")
    splits = pd.read_csv(OUTPUT_DIR / "splits.csv")
    report = (OUTPUT_DIR / "classification_report.txt").read_text()
    cm = m["confusion_matrix [[TN,FP],[FN,TP]] (Fresh=0, Rotten=1)"]

    sizes = splits["split"].value_counts()
    lines = []
    a = lines.append

    a("_These numbers were generated automatically from `outputs/metrics.json` "
      "by `python -m src.make_report` after running `python -m src.train`._\n")
    a(f"**Data used:** {len(splits)} unique images "
      f"(train {sizes.get('train', 0)}, validation {sizes.get('val', 0)}, "
      f"test {sizes.get('test', 0)}). Exact duplicate files were excluded so that no "
      f"image can appear in more than one split.\n")
    a(f"**Chosen on validation data only:** sigma = {m['chosen_sigma']:.4g}, "
      f"priors = `{m['chosen_priors']}` "
      f"(validation macro-F1 = {m['val_macro_f1_at_chosen']:.3f}).\n")

    a("### Test-set performance (unseen images)\n")
    a("| Metric | Value |\n|---|---|")
    a(f"| Accuracy | {m['accuracy']:.3f} |")
    a(f"| Precision (macro) | {m['precision_macro']:.3f} |")
    a(f"| Recall (macro) | {m['recall_macro']:.3f} |")
    a(f"| F1-score (macro) | {m['f1_macro']:.3f} |")
    a(f"| Precision / Recall / F1 for Rotten | "
      f"{m['precision_rotten']:.3f} / {m['recall_rotten']:.3f} / {m['f1_rotten']:.3f} |")
    a(f"| ROC-AUC (Rotten = positive) | {m['roc_auc']:.3f} |")
    a(f"| Fruit-type PNN accuracy | {m['fruit_type_accuracy_test']:.3f} |")
    a(f"| Test images | {int(m['n_test'])} |\n")

    a("### Confusion matrix (rows = actual, columns = predicted)\n")
    a(f"| | Pred {CLASS_NAMES[0]} | Pred {CLASS_NAMES[1]} |\n|---|---|---|")
    a(f"| **Actual {CLASS_NAMES[0]}** | {cm[0][0]} | {cm[0][1]} |")
    a(f"| **Actual {CLASS_NAMES[1]}** | {cm[1][0]} | {cm[1][1]} |\n")

    a("### Classification report\n")
    a("```\n" + report.rstrip() + "\n```\n")

    a("### Per-fruit performance (test set)\n")
    a("| Fruit | Test images | Fresh | Rotten | Accuracy | Rotten precision | "
      "Rotten recall | Rotten F1 | Errors |\n|---|---|---|---|---|---|---|---|---|")
    for _, r in per_fruit.iterrows():
        a(f"| {r['fruit']} | {int(r['n_test'])} | {int(r['n_fresh'])} | "
          f"{int(r['n_rotten'])} | {r['accuracy']:.3f} | {r['rotten_precision']:.3f} | "
          f"{r['rotten_recall']:.3f} | {r['rotten_f1']:.3f} | {int(r['errors'])} |")
    a("\nEach fruit has very few test images, so these per-fruit figures are noisy: "
      "one image changes them by several percentage points.\n")

    a("### Confidence versus calibration (measured)\n")
    wrong = m["mean_confidence_when_wrong"]
    a(f"* Mean confidence on the test set: **{m['mean_confidence']:.3f}**, "
      f"while accuracy was **{m['accuracy']:.3f}**.")
    a(f"* Mean confidence when the prediction was correct: "
      f"{m['mean_confidence_when_correct']:.3f}"
      + (f"; when wrong: {wrong:.3f}." if wrong is not None else "."))
    a(f"* Brier score: {m['brier_score']:.3f} (lower is better; 0 is perfect).")
    gap = m["mean_confidence"] - m["accuracy"]
    if gap > 0.03:
        a("* Mean confidence is higher than accuracy, so the scores are "
          "**over-confident** and must not be read as calibrated probabilities.")
    elif gap < -0.03:
        a("* Mean confidence is lower than accuracy, so the scores are "
          "**under-confident** and must not be read as calibrated probabilities.")
    else:
        a("* Mean confidence is close to accuracy on this test set, but that is not "
          "proof of calibration; no calibration step was applied.")
    a("")

    a("### Plots\n")
    a("Saved in `outputs/`: `confusion_matrix.png`, `roc_curve.png`, `sigma_tuning.png`.\n")
    return "\n".join(lines)


def main():
    block = f"{START}\n\n{build_results_markdown()}\n{END}"
    text = README.read_text(encoding="utf-8") if README.exists() else ""
    if START in text and END in text:
        head = text.split(START)[0]
        tail = text.split(END, 1)[1]
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n## Results\n\n" + block + "\n"
    README.write_text(text, encoding="utf-8")
    print("README.md results section updated from outputs/.")


if __name__ == "__main__":
    main()