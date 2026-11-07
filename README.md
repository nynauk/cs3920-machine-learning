# CS3920 Machine Learning: nearest neighbours, SVMs and conformal prediction

My coursework and lab work for CS3920 Machine Learning at Royal Holloway (autumn 2025). The thread running through
it is conformal prediction: not just predicting a label, but attaching a p-value to every candidate label so the
classifier can say how sure it is, with a guarantee that it is wrong at most a chosen fraction of the time.

> If you are taking CS3920, please do not copy this. It is here as a record of my own work.

| Folder | What it is |
|---|---|
| [`cw1-nearest-neighbour-conformal/`](cw1-nearest-neighbour-conformal) | Assignment 1: 1-NN, k-NN and a nearest-neighbour conformal predictor written from scratch in NumPy |
| [`cw2-svm-conformal/`](cw2-svm-conformal) | Assignment 2: support vector machines with scaling pipelines, grid-searched hyperparameters and a cross-conformal predictor |
| [`labs/`](labs) | Worksheet 6 (scaling and Lasso regression on the diabetes data) and worksheet 8 (neural networks and SVMs) |

## Assignment 1: nearest neighbours and conformal prediction

- **1-NN and k-NN from scratch.** Euclidean distance with a single pass over the training set per query; k-NN keeps
  the k smallest distances in a fixed array instead of sorting all of them, so each query is O(n) rather than
  O(n log n).
- **Conformal predictor.** The nonconformity score is the distance to the nearest example of the same class divided
  by the distance to the nearest example of a different class, with the edge cases defined explicitly
  (a/0 = infinity for a > 0, and 0/0 = 0). Each candidate label gets the p-value
  (number of training scores at least as large as the test score, plus one) / (n + 1).
- **Validity check.** A plot of the fraction of true-label p-values at or below each significance level against the
  diagonal.

Stratified 75/25 split with seed 2025:

| Dataset | Train / test | 1-NN error | 3-NN error | Average false p-value |
|---|---|---:|---:|---:|
| Iris | 112 / 38 | 2.6% | 2.6% | 0.011 |
| Ionosphere | 263 / 88 | 15.9% | 17.0% | 0.066 |

The average false p-value is the largest p-value given to a wrong label, averaged over the test set; lower means
the predictor separates the classes more confidently. Ionosphere is harder, and the conformal predictor reflects
that by being less certain while staying valid.

## Assignment 2: SVMs and cross-conformal prediction

Two datasets with opposite characteristics: Wine (178 samples, 13 chemical measurements on very different scales)
and USPS handwritten digits (9,298 images of 16x16 pixels, merged from the original train and test files).

- **Default SVM, 5-fold cross-validation on the training set.** Wine: 0.669 accuracy with a standard deviation of
  0.062 across folds, and 73.3% on the test set. USPS: 0.970 with a standard deviation of 0.003, and 97.7% on test.
- **Pipelines.** Each scaler (StandardScaler, MinMaxScaler, RobustScaler, Normalizer) is placed inside a pipeline
  with the SVC, and C and gamma are tuned by GridSearchCV on the training set only, so no test information leaks
  into the scaling or the search.
- **Cross-conformal predictor.** Five folds of the training set, using the SVM's decision function as the
  conformity score, with a calibration plot against the diagonal.

| Dataset | Best pipeline on test | Best C, gamma | Test accuracy (default SVM) | Cross-conformal average false p-value |
|---|---|---|---:|---:|
| Wine | StandardScaler + SVC | 1, 0.01 | 97.8% (73.3%) | 0.0085 |
| USPS | Normalizer + SVC | 10, 1 | 98.2% (97.7%) | 0.0056 |

The cross-conformal predictor was built on the tuned StandardScaler pipeline for both datasets.

The contrast is the main lesson. Wine's features are measured in different units, so the RBF kernel is dominated
by the largest ones until they are scaled; scaling plus tuning cut the test error from 26.7% to 2.2%. Per-sample
normalisation, which ignores feature scale, was the worst choice there (71.1%) but the best on USPS, where every
feature is a pixel on the same scale and what matters is the shape of the stroke, not its intensity.

## Labs

- **Worksheet 6:** Lasso regression on the 442-sample diabetes data, the effect of fitting a StandardScaler on the
  training data only, and choosing the regularisation strength by cross-validation.
- **Worksheet 8:** a multi-layer perceptron on the two-moons data, a linear SVM on the forge data, and an RBF SVM
  on the breast cancer data with and without scaling.

## Run

```bash
python -m pip install numpy scikit-learn matplotlib pandas jupyter
jupyter notebook
```

Data included: `ionosphere.txt` (UCI Ionosphere, CC BY 4.0), `zip.train.gz` and `zip.test.gz` (USPS digits, from
the datasets page of *The Elements of Statistical Learning*), `diabetes.data` (Efron et al., 2004) and the forge
sample points used in worksheet 8. Wine, Iris and breast cancer load from scikit-learn.
