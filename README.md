# Machine Learning Examples

Simple, editable examples using 10 rows of sample data. Each script can be run
independently and prompts for values to predict or transform.

## Requirements

```powershell
python -m pip install pandas matplotlib scikit-learn
```

## Examples

| File | Method | Example |
|---|---|---|
| `LinearRegression.py` | Linear regression | Predict house price from area |
| `MultivariateLinearRegression.py` | Multiple linear regression | Predict price from area and bedrooms |
| `LogisticRegression.py` | Logistic regression | Predict pass/fail |
| `DecisionTree.py` | Decision tree | Predict pass/fail and display the tree |
| `Bagging.py` | Bagging classifier | Predict pass/fail with combined trees |
| `Boosting.py` | Gradient boosting classifier | Predict pass/fail with sequential trees |
| `SVM.py` | Support vector machine | Predict pass/fail |
| `PCA.py` | Principal component analysis | Reduce features to two components |
| `GraphBasedClustering.py` | Spectral clustering | Group customers into clusters |

Run an example from PowerShell:

```powershell
python LogisticRegression.py
```

Edit the `data` dictionary in a script to use different example rows. Keep
feature columns, target values, and the prompted input fields consistent.
Classification scores are measured on a small held-out test set and are for
demonstration only. PCA is a dimensionality-reduction method, so it reports
explained variance rather than classification accuracy. Spectral clustering
does not natively assign clusters to new rows; its new-input cluster is an
approximation based on the nearest cluster center.
