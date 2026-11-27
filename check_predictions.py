
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_log_error, mean_absolute_error

# Load actual values and predictions
train = pd.read_csv("train.csv")
predictions = pd.read_csv("predictions.csv")

# Match predictions to their actual hours using project_id
results = predictions.merge(
    train[["project_id", "hours_spent"]],
    on="project_id",
    suffixes=("_predicted", "_actual")
)

actual = results["hours_spent_actual"]
predicted = results["hours_spent_predicted"]

# Calculate scores
rmsle = root_mean_squared_log_error(actual, predicted)
mae = mean_absolute_error(actual, predicted)

print("\n========== RESULTS ==========")
print(f"Projects checked: {len(results)}")
print(f"RMSLE: {rmsle:.5f}")
print(f"Mean absolute error: {mae:.2f} hours")

print("\n========== COMPARISON ==========")

results["difference"] = predicted - actual
results["error_percent"] = (
    (predicted - actual) / actual.clip(lower=0.01)
) * 100

print(
    results[[
        "project_id",
        "hours_spent_actual",
        "hours_spent_predicted",
        "difference",
        "error_percent"
    ]].round(2).to_string(index=False)
)
