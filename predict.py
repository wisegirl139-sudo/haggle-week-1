
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

print("Loading data...")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Same feature engineering used for V1
def engineer_features(df):
    df = df.copy()

    def safe_div(a, b):
        return a / (b + 1)

    df["num_commits_per_active_commit_days"] = safe_div(
        df["num_commits"], df["active_commit_days"]
    )
    df["lines_added_total_per_num_commits"] = safe_div(
        df["lines_added_total"], df["num_commits"]
    )
    df["lines_deleted_total_per_num_commits"] = safe_div(
        df["lines_deleted_total"], df["num_commits"]
    )
    df["active_commit_days_per_project_span_days"] = safe_div(
        df["active_commit_days"], df["project_span_days"]
    )
    df["lines_of_code_per_num_source_files"] = safe_div(
        df["lines_of_code"], df["num_source_files"]
    )
    df["num_source_files_per_num_directories"] = safe_div(
        df["num_source_files"], df["num_directories"]
    )
    df["num_dependencies_per_num_source_files"] = safe_div(
        df["num_dependencies"], df["num_source_files"]
    )
    df["readme_word_count_per_readme_num_headings"] = safe_div(
        df["readme_word_count"], df["readme_num_headings"]
    )
    df["readme_num_images_per_readme_word_count"] = safe_div(
        df["readme_num_images"], df["readme_word_count"]
    )
    df["asset_size_mb_per_num_image_files"] = safe_div(
        df["asset_size_mb"], df["num_image_files"]
    )
    df["num_test_files_per_num_source_files"] = safe_div(
        df["num_test_files"], df["num_source_files"]
    )

    df["commits_per_project_day"] = safe_div(
        df["num_commits"], df["project_span_days"]
    )
    df["total_code_churn"] = (
        df["lines_added_total"] + df["lines_deleted_total"]
    )
    df["code_growth_ratio"] = safe_div(
        df["lines_added_total"], df["lines_of_code"]
    )
    df["non_merge_commits"] = (
        df["num_commits"] - df["num_merge_commits"]
    )

    return df


print("Preparing features...")

# Remove labels and random identifier
drop_cols = ["hours_spent", "log_hours", "hours_bucket", "project_id"]

X_train = engineer_features(train.drop(columns=drop_cols, errors="ignore"))
X_test = engineer_features(test.drop(columns=drop_cols, errors="ignore"))

# Reproduce V1 categorical preprocessing
cat_cols = X_train.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

for col in cat_cols:
    X_train[col] = X_train[col].fillna("__MISSING__").astype(str)
    X_test[col] = X_test[col].fillna("__MISSING__").astype(str)

# Load V1 model
print("Loading V1 model...")

model = CatBoostRegressor()
model.load_model("hours_model.cbm")

# Match the exact feature order stored in the model
expected_features = model.feature_names_

missing = [c for c in expected_features if c not in X_test.columns]

if missing:
    raise ValueError(f"Missing model features: {missing}")

X_test = X_test[expected_features]

print("Predicting...")

# V1 was trained on log1p(hours_spent)
log_predictions = model.predict(X_test)
predictions = np.expm1(log_predictions)

# Keep predictions within the competition's target range
predictions = np.clip(predictions, 0, 140.7)

# Save submission
submission = pd.DataFrame({
    "project_id": test["project_id"],
    "hours_spent": predictions
})

submission.to_csv("predictions.csv", index=False)

print("\nSUCCESS!")
print("Predictions saved to predictions.csv")
print("Number of predictions:", len(submission))
print("\nFirst predictions:")
print(submission.head())
