# FruitFresh AI: Fruit Freshness Classification with a Probabilistic Neural Network (PNN)

An academic mini project that classifies fruit images as **Fresh** or **Rotten**
using a **Probabilistic Neural Network implemented from scratch in NumPy**. It shows
a confidence score for every prediction through a Streamlit web app.

> The classifier is a genuine PNN (Specht, 1990). No CNN, ANN, SVM, KNN or Random
> Forest is used for classification. scikit-learn is used only for the train/test
> split and for evaluation metrics.

---

## 1. Dataset

Kaggle: *Fruit Freshness Classification (Fresh vs Rotten)*, 566 images of
Apple, Banana and Strawberry (License: Apache 2.0).

| Fruit | Fresh | Rotten |
|---|---|---|
| Apple | 124 | 55 |
| Banana | 33 | 90 |
| Strawberry | 220 | 44 |
| **Total** | **377** | **189** |

Checks performed by `src/preprocessing.py` (the original images are never modified,
moved or deleted):

* All images open correctly (no corrupted files found in this dataset).
* The dataset contains **44 exact duplicate files** (byte-identical copies), leaving
  **522 unique images**. Duplicates are excluded from the model's data so that a
  copy of a training image can never appear in the test set (data leakage).
* Formats are mixed (JPG, PNG, WEBP, JFIF), and some PNGs are transparent. Everything
  is converted to 3-channel RGB.

**Splitting:** 70% train / 15% validation / 15% test, stratified on fruit and class
together, with a fixed random seed. Assertions in the code stop the run if any image
appears in two splits.

## 2. Project structure

```
Fruit_Freshness_Classification/
├── dataset/                  original images (read-only use)
├── models/                   saved PNN + scaler (pnn_model.joblib)
├── outputs/                  split file, metrics, plots, predictions
├── src/
│   ├── preprocessing.py      dataset checks, image loading, leak-free split
│   ├── feature_extraction.py colour / texture / shape features
│   ├── pnn.py                the Probabilistic Neural Network (+ feature scaler)
│   ├── train.py              tuning on validation, evaluation on test, saving
│   ├── predict.py            single-image prediction
│   └── make_report.py        writes real results into this README
├── app.py                    Streamlit app "FruitFresh AI"
├── main.py                   command-line entry point
├── requirements.txt
└── README.md
```

## 3. Installation (Windows, VS Code)

```powershell
cd Fruit_Freshness_Classification
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

## 4. How to run

```powershell
python -m src.preprocessing      # verify dataset, create train/val/test split
python -m src.train              # tune sigma on validation, evaluate on test, save model
python -m src.make_report        # write the real results into this README
streamlit run app.py             # launch FruitFresh AI in the browser
```

Predict one image from the terminal: `python -m src.predict path\to\image.jpg`

Equivalent shortcuts: `python main.py check | train | predict <image> | app`.

## 5. Image preprocessing and features

Every image is loaded with Pillow, converted to RGB, and resized to 128 x 128
(`INTER_AREA`, the best OpenCV method for shrinking). A 468-value feature vector is then
built from four blocks:

| Block | Size | Why it is useful |
|---|---|---|
| HSV colour histogram (8x4x4 bins, normalised) | 128 | Hue separates colour from brightness; rotten fruit shifts towards brown and dark |
| Lab colour moments (mean and std of L, a, b) | 6 | Compact brightness/colour cues matching human perception |
| HOG (9 orientations, 16x16 cells, on a 64x64 grayscale copy) | 324 | Edge and shape structure; wrinkles and bruises |
| LBP texture histogram (uniform, 8 points) | 10 | Fine surface texture: spots, mould, wrinkling |

**Normalisation:** each feature is standardised using the **training set only**
(mean 0, std 1). Each block is then divided by the square root of its size so the
324-value HOG block does not outweigh the 6-value Lab block in distance calculations.
The scaler is saved and re-used by the app.

## 6. The Probabilistic Neural Network

A PNN does not learn weights by gradient descent. "Training" means **storing the
training patterns**. Classification then follows Bayes' decision rule using Parzen
(Gaussian-kernel) density estimates.

| Layer | Computation |
|---|---|
| 1. Input | Receives the feature vector `x` (d = 468 values) and passes it to every pattern neuron |
| 2. Pattern | One neuron per training sample `x_i`: `k_i(x) = exp( -||x - x_i||^2 / (2 sigma^2) )` |
| 3. Summation | One neuron per class `c`: `p(x given c) = 1 / (N_c (2 pi)^(d/2) sigma^d) * sum over i in class c of k_i(x)` |
| 4. Output | Bayes' rule: `P(c given x) = pi_c p(x given c) / sum over k of pi_k p(x given k)`, then predict the class with the highest posterior |

* **sigma** is the smoothing parameter. A small sigma lets each training image influence
  only its close neighbourhood (can over-fit); a large sigma blurs the classes together
  (can under-fit). It is tuned on the **validation set only**.
* **Numerical safety:** with 468 features the kernel values underflow to 0 in ordinary
  floating point, so all computations are done in log space with the log-sum-exp trick.
  The mathematics is unchanged. The code was checked against a plain textbook
  implementation on small data and agreed to about 1e-16.

### Class imbalance

* Each class density is divided by its own sample count `N_c`, so the density estimates are
  not inflated for the larger class.
* The imbalance therefore enters only through the **priors** `pi_c`. Two options are
  implemented: `empirical` (training class ratio, favours the majority class) and
  `uniform` (equal priors). The better option is chosen on validation macro-F1
  together with sigma.
* Model selection uses **macro-F1**, which weights Fresh and Rotten equally, instead of
  accuracy, which the majority class can dominate.
* Stratified splitting keeps the Fresh/Rotten ratio inside every fruit in every split.

### Confidence score (important)

The app shows the **normalised class posteriors** `P(Fresh | x)` and `P(Rotten | x)`, and the
larger one is the "model confidence". These are **model scores, not statistically calibrated
probabilities.** A calibrated 90% would mean that about 9 in 10 predictions with that score
are correct. PNN posteriors in high dimensions are pushed towards 0 or 1, so they tend to
look more certain than the model really is. The Results section below reports the measured
gap between mean confidence and accuracy on the test set. No calibration method was applied.

The app also shows a "borderline" note when the winning score is below 70%, and an
"unusual image" warning when the upload is much farther from every training image than
training images are from each other. This warning is a simple heuristic and does not catch
every non-fruit image.

### Fruit type

The fruit type shown in the app comes from a **second PNN** trained on the same features
with the labels Apple/Banana/Strawberry. It is an estimate, and its test accuracy is
reported below.

## 7. Evaluation protocol

1. Features and scaler are fitted on **train** only.
2. sigma and priors are chosen on **validation** only.
3. The **test** set is used once, after those choices are fixed.
4. All numbers below are computed by the code and saved to `outputs/`. None are typed in.

## 8. Results

<!-- RESULTS START -->

_These numbers were generated automatically from `outputs/metrics.json` by `python -m src.make_report` after running `python -m src.train`._

**Data used:** 522 unique images (train 364, validation 79, test 79). Exact duplicate files were excluded so that no image can appear in more than one split.

**Chosen on validation data only:** sigma = 0.2956, priors = `uniform` (validation macro-F1 = 0.959).

### Test-set performance (unseen images)

| Metric | Value |
|---|---|
| Accuracy | 0.861 |
| Precision (macro) | 0.847 |
| Recall (macro) | 0.852 |
| F1-score (macro) | 0.849 |
| Precision / Recall / F1 for Rotten | 0.793 / 0.821 / 0.807 |
| ROC-AUC (Rotten = positive) | 0.923 |
| Fruit-type PNN accuracy | 0.785 |
| Test images | 79 |

### Confusion matrix (rows = actual, columns = predicted)

| | Pred Fresh | Pred Rotten |
|---|---|---|
| **Actual Fresh** | 45 | 6 |
| **Actual Rotten** | 5 | 23 |

### Classification report

```
              precision    recall  f1-score   support

       Fresh      0.900     0.882     0.891        51
      Rotten      0.793     0.821     0.807        28

    accuracy                          0.861        79
   macro avg      0.847     0.852     0.849        79
weighted avg      0.862     0.861     0.861        79
```

### Per-fruit performance (test set)

| Fruit | Test images | Fresh | Rotten | Accuracy | Rotten precision | Rotten recall | Rotten F1 | Errors |
|---|---|---|---|---|---|---|---|---|
| Apple | 25 | 17 | 8 | 0.920 | 1.000 | 0.750 | 0.857 | 2 |
| Banana | 18 | 5 | 13 | 0.778 | 0.765 | 1.000 | 0.867 | 4 |
| Strawberry | 36 | 29 | 7 | 0.861 | 0.667 | 0.571 | 0.615 | 5 |

Each fruit has very few test images, so these per-fruit figures are noisy: one image changes them by several percentage points.

### Confidence versus calibration (measured)

* Mean confidence on the test set: **0.952**, while accuracy was **0.861**.
* Mean confidence when the prediction was correct: 0.966; when wrong: 0.865.
* Brier score: 0.112 (lower is better; 0 is perfect).
* Mean confidence is higher than accuracy, so the scores are **over-confident** and must not be read as calibrated probabilities.

### Plots

Saved in `outputs/`: `confusion_matrix.png`, `roc_curve.png`, `sigma_tuning.png`.

<!-- RESULTS END -->

## 9. Limitations and discussion

* **Small dataset.** Only 522 unique images, with a test set of about 79. One image changes
  accuracy by roughly 1.3 percentage points, so results are uncertain; a different random
  split would give somewhat different numbers.
* **Class imbalance.** Rotten Strawberry (44 images) and Fresh Banana (33 images) are rare.
  Their test counts are only a handful, so the per-fruit figures for these groups are very
  noisy. Uniform priors and macro-F1 reduce the bias towards the majority class but cannot
  create information that is missing from the data.
* **Validation versus test.** The chosen sigma is the best of many candidates on a small
  validation set, so the validation score is slightly optimistic. The test score is the one
  to quote.
* **Near-duplicates.** Only byte-identical files were removed. The images were collected from
  public web sources, so visually similar photos (the same fruit in slightly different
  crops) may exist across splits and could make results somewhat optimistic.
* **Hand-crafted features.** Colour, HOG and LBP features are sensitive to background,
  lighting and camera. Real photos that differ from the dataset's style may be misclassified.
* **PNN cost and scope.** A PNN stores every training sample and compares each new image
  with all of them, which scales poorly to large datasets. The model knows only apples,
  bananas and strawberries; it will still output a score for any other object.
* **Scores are not calibrated** (see section 6).

## 10. Possible improvements

More images per class and per fruit, near-duplicate detection, data augmentation, cross-validation
for more stable estimates, per-feature-block sigma tuning, and a calibration step
(for example temperature scaling fitted on validation data).

## 11. Reference

D. F. Specht, "Probabilistic neural networks," *Neural Networks*, vol. 3, no. 1,
pp. 109-118, 1990.