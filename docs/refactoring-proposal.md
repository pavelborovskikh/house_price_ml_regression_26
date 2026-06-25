# Refactoring Proposal

This document captures structural and code-quality improvements identified during a review of the project.

---

## 1. Restructure the repository

Currently everything lives flat at the root. Separating concerns into subdirectories makes the project easier to navigate and extend.

**Proposed layout:**

```
house_price_ml_regression_26/
├── src/
│   ├── features.py         # shared feature-name constants
│   ├── train.py
│   ├── predict.py
│   └── app.py
├── data/
│   └── kc_house_data.csv
├── models/                 # gitignored — generated at training time
│   ├── model.pkl
│   └── preprocessor.pkl
├── notebooks/
│   └── notebook.ipynb
├── tests/
│   ├── test.py
│   └── test_api.py
├── docs/
│   └── refactoring-proposal.md   (this file)
├── Dockerfile
├── fly.toml
├── pyproject.toml
├── uv.lock
└── README.md
```

Add `models/` to `.gitignore` so generated artifacts are never committed.

---

## 2. Eliminate duplicated feature constants

`NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` are defined identically in both `train.py` and `predict.py`. If either list is updated in one file but not the other, the model will be trained on different features than those sent at inference time — a silent bug that is hard to catch.

**Fix:** extract both lists into a single `src/features.py` and import from there in both files.

```python
# src/features.py
NUMERIC_FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "sqft_above", "sqft_basement", "yr_built", "lat", "long",
    "sqft_living15", "sqft_lot15", "house_age", "renovated",
]
CATEGORICAL_FEATURES = ["waterfront", "view", "condition", "grade", "zipcode"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
```

---

## 3. Remove the redundant preprocessor artifact

`train.py` saves both `model.pkl` and `preprocessor.pkl`, but the sklearn `Pipeline` already embeds the preprocessor as its first step. `predict.py` loads `model.pkl` and never uses `preprocessor.pkl`. The second file is dead weight and a source of confusion.

**Fix:** delete `save_model_and_preprocessor`'s preprocessor save, and remove the `PREPROCESSOR_PATH` / `preprocessor.pkl` references throughout.

---

## 4. Fix stale model description in the API

`app.py`'s `/health` and `/info` endpoints hardcode `"Random Forest"` in their responses, but training dynamically selects the best model — which is currently XGBoost. Any consumer of these endpoints gets incorrect metadata.

**Fix:** pass the model type as a string when initialising `PredictionService`, derived from the actual best model selected in `train.py`, and surface it from `PredictionService.get_model_type()`.

```python
# in PredictionService
def get_model_type(self) -> str:
    return type(self.model.named_steps["model"]).__name__
```

---

## 5. Fix the `house_age` leakage / inconsistency

`train.py` computes `house_age = 2015 - yr_built` (hardcoded to the dataset year). The API then accepts `house_age` as a caller-supplied field, meaning:

- Training always uses ages relative to 2015.
- A caller in 2026 who supplies a current age will produce a feature value the model was never trained on.

**Options (pick one):**

- **Compute server-side:** remove `house_age` from `HousePredictionRequest`, derive it inside `PredictionService.predict()` using the same `2015 - yr_built` formula, and document the convention.
- **Make it explicit:** keep `house_age` in the request but add a note in the README and the Pydantic field description that callers must pass `2015 - yr_built`, not the actual current age.

The first option is safer because it removes the ambiguity entirely.
