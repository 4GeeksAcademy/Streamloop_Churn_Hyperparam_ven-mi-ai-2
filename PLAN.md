# StreamLoop churn tuning plan

Challenge source: `StreamLoop_Hyperparams.md`.

StreamLoop already has a first churn classifier. This pass does not change the question the model answers. It shows that the configuration was searched, that the search never saw the test set, and that the final choice is explained with the business metric and the fold-to-fold spread.

## Business metric

A missed churn (false negative) costs more than a retention offer sent to a customer who would have stayed (false positive). Accuracy is the wrong objective: about three quarters of customers stay, so a model that never flags anyone looks strong and catches nobody.

Primary score, used for every search and for the baseline-versus-tuned comparison:

- **F2** (`fbeta_score` with `beta=2`). Recall is weighted four times as heavily as precision in the F-beta trade-off, so the search is pushed toward catching churners without treating "flag everyone" as a perfect model.

Reported alongside F2, on the test set only at the two allowed moments, so the comparison is complete:

- recall, precision, F1, ROC AUC, accuracy

Accuracy is recorded so the business point is visible. It is not the search objective.

Decision threshold stays at 0.5. Threshold search is a separate pass and is out of scope here.

## Classifier

`RandomForestClassifier`. Model family is not part of the grade. A forest has a real hyperparameter surface (depth, leaf size, feature subsample, class weight) and trains comfortably on this table.

`random_state=42` is fixed on every forest so reruns match. That is a reproducibility control, not a tuned value. Every other baseline setting is the scikit-learn default (`n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `class_weight=None`).

Trees are invariant to monotone rescaling, so the pipeline does not scale numeric columns. Imputation and one-hot encoding stay, because those steps change the information the forest sees.

## Data

URL (loaded in code, not saved by hand):

`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

Target: `Churn`, mapped with a fixed rule `Yes -> 1`, `No -> 0`.

### Minimal cleaning before the split

These steps are row-local or schema fixes. They do not estimate a statistic from the sample.

- Drop `customerID`. It identifies a row and is not a predictor.
- Coerce `TotalCharges` to float. Blank strings become missing.
- Where `TotalCharges` is missing and `tenure == 0`, set `TotalCharges` to 0. Those are new accounts, not unknown bills. Any other missing `TotalCharges` is left missing for the imputer inside the pipeline.

### Learned preprocessing, only inside the pipeline

Fit on training folds only, because the transformers sit inside the `Pipeline` that `fit` / `*SearchCV` clones.

- Numeric (`tenure`, `MonthlyCharges`, `TotalCharges`, `SeniorCitizen`): median `SimpleImputer`.
- All remaining columns: most-frequent `SimpleImputer`, then `OneHotEncoder(handle_unknown="ignore")`.

`SeniorCitizen` is already 0/1 and stays numeric.

## Split policy

Immediately after the cleaning above, and before any pipeline is fit:

- `train_test_split`, `test_size=0.2`, `stratify=y`, `random_state=42`.

The test set is scored exactly twice:

1. Default-hyperparameter pipeline, after it is fit on the training split.
2. The one selected tuned pipeline, after search is finished.

`RandomizedSearchCV` and `GridSearchCV` are fit on the training split only. Nothing in the search is fit on the full dataset. `refit=True` retrains the winning parameters on the training split. That object is not fit a second time if it is the model we keep.

Cross-validation is `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, shared by both searches so the folds mean the same thing.

## Baseline

1. Build the pipeline with a default forest (plus `random_state`).
2. Fit on the training split.
3. Score the test split once. Store F2, recall, precision, F1, ROC AUC, and accuracy.
4. Only then build the searches.

## Broad search, then a narrow one

Search scoring is the F2 scorer. Both searches use `n_jobs=-1`, the shared stratified folds, and `refit=True`.

### Random search space

Each list is what that forest argument actually accepts. The grid is the cartesian product of these lists (2,160 combinations). The random search draws 24 of them (`n_iter=24`, `random_state=42`).

| Parameter | Values |
| --- | --- |
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 4, 8, 12, 20, `None` |
| `min_samples_split` | 2, 5, 10, 20 |
| `min_samples_leaf` | 1, 2, 4, 8 |
| `max_features` | `"sqrt"`, `"log2"`, 0.5 |
| `class_weight` | `None`, `"balanced"`, `"balanced_subsample"` |

Pipeline keys are prefixed `model__`.

### How the random-search result narrows the grid

Locked before looking at scores:

1. Rank random-search trials by mean CV F2.
2. Take the top 5 trials as the promising region.
3. Unordered arguments (`max_features`, `class_weight`): if the best trial's value appears in at least 3 of those 5, freeze it. Otherwise keep every distinct value that appears in the top 5, always including the best value.
4. Ordered arguments (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`): keep the best value and its immediate neighbors in the table above (one step either side). This is a local refine, not a second full sweep.
5. If the cartesian product is larger than 36, freeze the ordered argument with the widest grid down to its single best value, and repeat until the product is at most 36.

Then run `GridSearchCV` on that grid, training split only, same folds, same F2 scorer, `refit=True`, `n_jobs=-1`.

## Choosing the final model

`cv_results_` from the grid is inspected for the top candidates: mean F2, fold standard deviation, the five fold scores, and the range across folds. Rank is not decided from the mean alone.

Selection rule, also locked before test scores exist:

1. Near-tie band: every grid candidate whose mean CV F2 is within 0.01 of the best mean.
2. Inside that band, prefer the lowest fold standard deviation. If two are equal on spread, prefer the higher mean.
3. If that choice is the `refit=True` winner, ship `best_estimator_` and do not fit it again.
4. If that choice is a different candidate, fit one new pipeline with those parameters on the training split only. Do not call `fit` on `best_estimator_` again. The test set is still unused until the final score.
5. If the tighter model sits more than 0.01 F2 below the best mean, it is outside the band. Ship the refit winner and say so, including its fold spread.

The written reason has to say which arm of this rule fired and what was given up (a little mean, or a little stability).

## Final test comparison

Score the selected pipeline on the test set once, with the same six metrics as the baseline. Compare tuned minus baseline on each metric. Do not run another search after seeing the test numbers.

## What this pass will not do

- No second model family, no stacking, no threshold sweep.
- No extra customer features beyond the cleaning above.
- No fit of either search on the full 7,043 rows.
- No third look at the test labels.

## Files

| File | Role |
| --- | --- |
| `PLAN.md` | This plan. |
| `streamloop_churn_tuning.py` | Load, split, baseline, both searches, selection, one final test score. |
| `requirements.txt` | `pandas` and `scikit-learn`. |
| `artifacts/run_results.json` | Machine-readable record of the run (metrics, parameters, fold table, which selection rule fired). |

`tuning_report.md` is the evaluation write-up. It is not written until the results have been reviewed and the save is approved.

## Evaluation checklist this implementation has to pass

Mapped from "What we will evaluate" in the challenge:

1. Preprocessing that is estimated from data lives in the `Pipeline`, and the split happens before any `fit`.
2. A default-hyperparameter baseline is fit and scored on the test set before either search starts.
3. `RandomizedSearchCV` explores the broad space, and only then `GridSearchCV` runs on the narrowed grid.
4. Both searches use stratified cross-validation, `n_jobs=-1`, and the training split only.
5. `scoring` is F2, with the cost story above. It is not accuracy.
6. Grid `cv_results_` is read for fold variance, not only the top mean.
7. The shipped parameters are the ones the selection rule produced, and the rule is stated.
8. Baseline and tuned test numbers use the same metrics.
