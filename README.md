# Nearest-neighbour classification and conformal prediction from scratch

Coursework for CS3920 Machine Learning at Royal Holloway (November 2025): a 1-NN and k-NN classifier and a
nearest-neighbour conformal predictor, written from scratch in NumPy without scikit-learn's models, and tested on
the Iris and Ionosphere datasets.

> If you are taking CS3920, please do not copy this. It is here as a record of my own work.

## What is implemented

- **1-NN and k-NN.** Euclidean distance with a single pass over the training set per query. k-NN keeps the k
  smallest distances in a fixed array instead of sorting all of them, so each query costs O(n) rather than
  O(n log n).
- **Conformal predictor.** The nonconformity score of a labelled point is the distance to the nearest example of
  the same class divided by the distance to the nearest example of a different class. Edge cases are defined
  explicitly: a/0 = infinity for a > 0, and 0/0 = 0. For each test point and each candidate label the p-value is
  the fraction of training scores at least as large as the test score, counting the test point itself.
- **Evaluation.** Test error rates for 1-NN and 3-NN, the average false p-value (the largest p-value given to a
  wrong label, averaged over the test set; lower means a more informative predictor), and a validity plot of the
  fraction of true-label p-values at or below each significance level against the diagonal.

## Results

Stratified 75/25 split with a fixed seed of 2025. The submitted version used a personal seed, so its numbers
differ slightly.

| Dataset | Train / test | 1-NN errors | 1-NN error rate | 3-NN error rate | Average false p-value |
|---|---|---:|---:|---:|---:|
| Iris | 112 / 38 | 1 | 2.6% | 2.6% | 0.011 |
| Ionosphere | 263 / 88 | 14 | 15.9% | 17.0% | 0.066 |

Iris is close to separable, so the predictor is both accurate and confident. On Ionosphere the nearest neighbour
is wrong about one time in six and the false p-values are about six times higher, which is what a conformal
predictor should do: it stays valid but becomes less certain when the data is harder.

## Run

```bash
python -m pip install numpy scikit-learn matplotlib jupyter
jupyter notebook nn_conformal_prediction.ipynb
```

`ionosphere.txt` is the UCI Ionosphere dataset (Sigillito et al., 1989; UCI Machine Learning Repository,
CC BY 4.0): 351 radar returns, 34 features, labelled good (1) or bad (-1).
