"""
pnn.py
------
A genuine Probabilistic Neural Network (Specht, 1990), written from scratch
with NumPy. No scikit-learn classifier is used anywhere in this file.

THE FOUR LAYERS
===============
1. INPUT LAYER     Receives the feature vector x (d numbers). No computation;
                   it just distributes x to every pattern neuron.

2. PATTERN LAYER   One neuron per TRAINING sample x_i. Each neuron measures how
                   similar x is to its stored sample with a Gaussian kernel:

                       k_i(x) = exp( -||x - x_i||^2 / (2 sigma^2) )

                   close to 1 when x is near x_i, close to 0 when it is far.

3. SUMMATION LAYER One neuron per CLASS c. It averages the kernels of that
                   class's pattern neurons. This is the Parzen-window estimate
                   of the class-conditional probability density:

                       p(x | c) = 1 / ( N_c (2*pi)^(d/2) sigma^d )
                                  * sum_{i in class c} k_i(x)

4. OUTPUT LAYER   Applies Bayes' rule with class priors pi_c:

                       P(c | x) = pi_c p(x | c) / sum_k pi_k p(x | k)

                   and picks the class with the highest posterior.

sigma (the smoothing parameter) is the only thing "learned" by tuning:
small sigma -> each training point only influences its immediate surroundings
(can over-fit); large sigma -> blurry densities (can under-fit).

NUMERICAL NOTE
==============
With ~468 features the kernel values (e.g. exp(-300)) underflow to 0 in normal
floating point. So everything is computed in LOG space and combined with the
log-sum-exp trick. The mathematics is identical, only numerically safe.
"""

import numpy as np


# ----------------------------------------------------------------------------
# small numeric helpers
# ----------------------------------------------------------------------------
def _logsumexp(a, axis=-1):
    """log(sum(exp(a))) computed without overflow/underflow."""
    a_max = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(a_max, axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis))


def _squared_distances(A, B):
    """
    Squared Euclidean distance between every row of A and every row of B.
    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b  (fast, fully vectorised).
    Result shape: (len(A), len(B)).
    """
    d2 = (np.sum(A ** 2, axis=1)[:, None]
          + np.sum(B ** 2, axis=1)[None, :]
          - 2.0 * A @ B.T)
    return np.maximum(d2, 0.0)          # tiny negatives from rounding -> 0


# ----------------------------------------------------------------------------
# feature scaling that keeps every feature block equally important
# ----------------------------------------------------------------------------
class BlockScaler:
    """
    Step 1: standardise every feature (subtract train mean, divide by train std)
            so no feature dominates just because it has bigger numbers.
    Step 2: divide each feature BLOCK (HSV, Lab, HOG, LBP) by sqrt(block size),
            so a 324-value HOG block does not outweigh the 6-value Lab block
            in the distance calculation.
    It is fitted on TRAINING data only (no leakage from val/test).
    """

    def __init__(self, block_slices):
        self.block_slices = block_slices

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-8] = 1.0           # constant features: avoid divide by 0
        self.std_ = std
        self.block_weight_ = np.ones(X.shape[1])
        for sl in self.block_slices.values():
            self.block_weight_[sl] = 1.0 / np.sqrt(sl.stop - sl.start)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_ * self.block_weight_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ----------------------------------------------------------------------------
# the PNN
# ----------------------------------------------------------------------------
class PNN:
    """
    Parameters
    ----------
    sigma : float
        Gaussian smoothing parameter (> 0).
    priors : "empirical" | "uniform" | sequence of floats
        "empirical": pi_c = N_c / N  (follows the training class ratio, so the
                     majority class is favoured).
        "uniform"  : pi_c = 1 / n_classes  (treats classes as equally likely;
                     this is how we correct for class imbalance).
        sequence   : your own priors, one per class (must sum to 1).
    """

    def __init__(self, sigma=1.0, priors="empirical"):
        self.sigma = sigma
        self.priors = priors

    # ---- training: a PNN just STORES the training patterns ------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        # Pattern layer memory: the training samples of each class
        self.patterns_ = {c: X[y == c] for c in self.classes_}
        self.class_counts_ = np.array(
            [len(self.patterns_[c]) for c in self.classes_], dtype=np.float64)
        return self

    def _class_priors(self):
        k = len(self.classes_)
        if isinstance(self.priors, str):
            if self.priors == "empirical":
                return self.class_counts_ / self.class_counts_.sum()
            if self.priors == "uniform":
                return np.full(k, 1.0 / k)
            raise ValueError("priors must be 'empirical', 'uniform' or a list")
        p = np.asarray(self.priors, dtype=np.float64)
        if p.shape != (k,) or not np.isclose(p.sum(), 1.0):
            raise ValueError("custom priors need one value per class, summing to 1")
        return p

    # ---- layers 1-3: log class-conditional densities --------------------------
    def log_densities(self, X):
        """
        Layers 1-3. Returns array (n_samples, n_classes) with log p(x | c).
        """
        X = np.asarray(X, dtype=np.float64)          # Layer 1: input
        d = self.n_features_
        s2 = self.sigma ** 2
        # log of the Gaussian normalising constant  (2*pi)^(d/2) * sigma^d
        log_norm = 0.5 * d * np.log(2.0 * np.pi) + d * np.log(self.sigma)

        out = np.empty((X.shape[0], len(self.classes_)))
        for j, c in enumerate(self.classes_):
            # Layer 2: pattern neurons -> log of each Gaussian kernel value
            log_kernels = -_squared_distances(X, self.patterns_[c]) / (2.0 * s2)
            # Layer 3: summation neuron -> average of kernels, then normalise
            out[:, j] = (_logsumexp(log_kernels, axis=1)
                         - np.log(self.class_counts_[j]) - log_norm)
        return out

    # ---- layer 4: Bayes decision ---------------------------------------------
    def predict_proba(self, X):
        """
        Layer 4. Posterior P(c | x) for each class (rows sum to 1).
        These are *normalised class posterior scores* of the model, NOT
        statistically calibrated probabilities.
        """
        log_post = self.log_densities(X) + np.log(self._class_priors())
        log_post -= _logsumexp(log_post, axis=1)[:, None]    # normalise
        return np.exp(log_post)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]