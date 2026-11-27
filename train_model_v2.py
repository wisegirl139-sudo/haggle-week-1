
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import root_mean_squared_log_error

# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("train.csv")

y = train["hours_spent"].clip(lower=0)
y_log = np.log1p(y)

X = train.drop(
    columns=["hours_spent", "log_hours",
             "hours_bucket", "project_id"],
    errors="ignore"
)

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def engineer_features(df):
    df = df.copy()

    def ratio(a, b):
        if a in df.columns and b in df.columns:
            df[a + "_per_" + b] = df[a] / (df[b] + 1)

    ratio("num_commits", "active_commit_days")
    ratio("lines_added_total", "num_commits")
    ratio("lines_deleted_total", "num_commits")
    ratio("active_commit_days", "project_span_days")
    ratio("lines_of_code", "num_source_files")
    ratio("num_source_files", "num_directories")
    ratio("num_dependencies", "num_source_files")
    ratio("readme_word_count", "readme_num_headings")
    ratio("readme_num_images", "readme_word_count")
    ratio("asset_size_mb", "num_image_files")
    ratio("num_test_files", "num_source_files")

    if "lines_added_total" in df and "lines_deleted_total" in df:
        df["total_code_churn"] = (
            df["lines_added_total"] + df["lines_deleted_total"]
        )

    if "lines_added_total" in df and "lines_of_code" in df:
        df["code_growth_ratio"] = (
            df["lines_added_total"] / (df["lines_of_code"] + 1)
        )

    if "num_commits" in df and "num_merge_commits" in df:
        df["non_merge_commits"] = (
            df["num_commits"] - df["num_merge_commits"]
        )

    return df.replace([np.inf, -np.inf], np.nan)


X = engineer_features(X)

cat_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for col in cat_features:
    X[col] = X[col].fillna("__MISSING__").astype(str)

print("Rows:", len(X))
print("Features:", X.shape[1])
print("Categorical features:", len(cat_features))

# ==========================================
# 3. CROSS VALIDATION
# ==========================================

bins = pd.qcut(
    y,
    q=5,
    labels=False,
    duplicates="drop"
)

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof = np.zeros(len(X))
scores = []
best_iterations = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(X, bins), 1
):

    print(f"\n========== FOLD {fold} ==========")

    model = CatBoostRegressor(
        iterations=4000,
        learning_rate=0.02,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=8,
        random_strength=1,
        rsm=0.9,
        random_seed=42 + fold,
        verbose=250,
        allow_writing_files=False
    )

    model.fit(
        X.iloc[train_idx],
        y_log.iloc[train_idx],
        cat_features=cat_features,
        eval_set=(
            X.iloc[val_idx],
            y_log.iloc[val_idx]
        ),
        early_stopping_rounds=200,
        verbose=250
    )

    pred = np.maximum(
        0,
        np.expm1(model.predict(X.iloc[val_idx]))
    )

    oof[val_idx] = pred

    score = root_mean_squared_log_error(
        y.iloc[val_idx],
        pred
    )

    scores.append(score)
    best_iterations.append(model.best_iteration_)

    print(f"Fold {fold} RMSLE: {score:.5f}")

# ==========================================
# 4. OVERALL RESULTS
# ==========================================

overall = root_mean_squared_log_error(y, oof)

print("\n========== V3 RESULTS ==========")

for i, score in enumerate(scores):
    print(f"Fold {i+1}: {score:.5f}")

print(f"\nMean RMSLE: {np.mean(scores):.5f}")
print(f"OOF RMSLE: {overall:.5f}")
print("Previous best: 0.65963")
print("Target: 0.50000")

pd.DataFrame({
    "actual": y,
    "predicted": oof
}).to_csv("v3_oof_predictions.csv", index=False)

# ==========================================
# 5. ERROR ANALYSIS
# ==========================================

print("\n========== ERROR BY PROJECT SIZE ==========")

groups = [
    ("Under 3 hours", 0, 3),
    ("3-6 hours", 3, 6),
    ("6-12 hours", 6, 12),
    ("12-30 hours", 12, 30),
    ("30+ hours", 30, float("inf"))
]

for name, low, high in groups:
    mask = (y >= low) & (y < high)

    score = root_mean_squared_log_error(
        y[mask],
        oof[mask]
    )

    print(f"{name}: {score:.5f} ({mask.sum()} projects)")

# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================

final_iterations = max(
    300,
    int(np.median(best_iterations))
)

final_model = CatBoostRegressor(
    iterations=final_iterations,
    learning_rate=0.02,
    depth=6,
    loss_function="RMSE",
    l2_leaf_reg=8,
    random_strength=1,
    rsm=0.9,
    random_seed=42,
    verbose=250,
    allow_writing_files=False
)

final_model.fit(
    X,
    y_log,
    cat_features=cat_features
)

final_model.save_model("hours_model_v3.cbm")

importance = pd.DataFrame({
    "feature": X.columns,
    "importance": final_model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv("v3_feature_importance.csv", index=False)

print("\nTOP 20 FEATURES")
print(importance.head(20).to_string(index=False))

print("\nV3 COMPLETE!")
