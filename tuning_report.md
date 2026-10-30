# StreamLoop churn model: tuning report

Code: `streamloop_churn_tuning.py`. Plan: `PLAN.md`. Raw numbers: `artifacts/run_results.json`.

## Setup

- Data: IBM Telco Customer Churn, loaded from the public URL. 7,043 customers, 26.5% churn.
- Cleaning before the split, with nothing estimated from the data: drop `customerID`, convert `TotalCharges` to a number, and set the 11 blank `TotalCharges` values to 0 (all were tenure-0 accounts).
- Split: stratified 80/20, `random_state=42`. 5,634 training rows and 1,409 test rows.
- Pipeline: a `ColumnTransformer` with a median imputer for numeric columns and a most-frequent imputer plus one-hot encoding for the rest, followed by `RandomForestClassifier(random_state=42)`. The imputers and the encoder are fit inside the pipeline, so cross-validation fits them only on training folds.
- Both searches use stratified 5-fold cross-validation on the training split only, with `n_jobs=-1` and `refit=True`.
- The test set was scored twice: once for the baseline and once for the final model.

## Metric choice

StreamLoop loses more from a churner it misses than from sending a retention offer to a customer who would have stayed. Accuracy hides that cost. With 73% of customers staying, a model that never flags anyone still looks accurate.

Search scoring is **F2** (`fbeta_score`, `beta=2`). F2 weights recall more heavily than precision, but it still penalizes a model that flags every customer. Recall, precision, F1, ROC AUC, and accuracy are reported alongside it for both models. The decision threshold stays at the default of 0.5.

## Baseline

Default forest: 100 trees, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight=None`. It was fit on the training split and scored on the test set before either search was built.

## Search

**Random search.** 24 draws from 2,160 combinations:

| Parameter | Values |
| --- | --- |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 4, 8, 12, 20, None |
| `min_samples_split` | 2, 5, 10, 20 |
| `min_samples_leaf` | 1, 2, 4, 8 |
| `max_features` | sqrt, log2, 0.5 |
| `class_weight` | None, balanced, balanced_subsample |

The best draw had a mean cross-validated F2 of 0.7205, with a fold standard deviation of 0.0169. The top five draws all used a class weight, mostly `max_features="log2"`, and shallow trees (depth 4 or 8). `max_features=0.5` and unlimited depth did worse.

**Grid search.** The grid kept the random-search winner plus one neighboring value on each side for each ordered parameter. `max_features` was frozen at log2 because it appeared in 4 of the top 5 draws. `min_samples_split` was frozen at 10 to keep the grid at 36 cells or fewer. The resulting 24-cell grid:

- `n_estimators`: 200, 300
- `max_depth`: 4, 8
- `min_samples_leaf`: 1, 2, 4
- `class_weight`: balanced, balanced_subsample
- `max_features`: log2
- `min_samples_split`: 10

## Fold variance and the final choice

The selection rule was fixed before the test score was known: keep every grid candidate whose mean F2 is within 0.01 of the best, then choose the one with the lowest fold standard deviation.

| Candidate | Depth | Leaf | Trees | Mean F2 | Fold std | Fold F2 scores |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Highest mean (`best_estimator_`) | 8 | 2 | 300 | 0.7256 | 0.0185 | 0.737, 0.697, 0.722, 0.752, 0.720 |
| **Shipped** | 4 | 1 | 300 | 0.7201 | 0.0169 | 0.718, 0.697, 0.716, 0.750, 0.719 |

Both candidates use `class_weight="balanced"`, `max_features="log2"`, and `min_samples_split=10`.

The shipped model gives up 0.0055 of mean F2 in exchange for 0.0016 less fold standard deviation. It is also a shallower forest, which is less likely to overfit as customer behavior drifts. The trade-off is small, and shipping the depth-8 model would also have been defensible. I kept the rule as fixed.

The shipped model is not `best_estimator_`. Its parameters were fit once on the training split, and `best_estimator_` was never refit.

**Final hyperparameters:** `n_estimators=300`, `max_depth=4`, `min_samples_split=10`, `min_samples_leaf=1`, `max_features="log2"`, `class_weight="balanced"`, `random_state=42`.

## Baseline vs. tuned, test set

| Metric | Baseline | Tuned | Change |
| --- | ---: | ---: | ---: |
| **F2** | 0.517 | **0.699** | +0.183 |
| Recall | 0.497 | 0.783 | +0.286 |
| Precision | 0.612 | 0.489 | -0.123 |
| F1 | 0.549 | 0.602 | +0.054 |
| ROC AUC | 0.823 | 0.835 | +0.012 |
| Accuracy | 0.783 | 0.725 | -0.057 |

The tuned model catches 78% of churners, up from 50%. In return, about half of its churn flags are customers who would have stayed. That is the trade the business asked for.

Most of the gain comes from changing the operating point, not from better ranking. ROC AUC rose only 0.012. The balanced class weight moves the model's 0.5 cutoff toward predicting churn. Accuracy fell, which is why it was not the objective.

Test F2 (0.699) sits about two points below the cross-validated F2 (0.720). That gap is within the range seen across folds (0.697 to 0.750).

## Next steps

- Tune the decision threshold on cross-validated predictions using real offer costs and churn losses. That would likely beat a fixed 0.5 cutoff.
- Repeat the search with a few different CV seeds to check whether the depth-4 versus depth-8 ranking holds up.
