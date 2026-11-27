
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Load our best model (V1)
model = CatBoostRegressor()
model.load_model("hours_model.cbm")

# Recreate the training features
X_test = test.drop(columns=["project_id"], errors="ignore")

# Predict hours
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, 140.7)

# Save submission
submission = pd.DataFrame({
    "project_id": test["project_id"],
    "hours_spent": predictions
})

submission.to_csv("predictions.csv", index=False)

print("Predictions saved!")
print(submission.head())
print("Total predictions:", len(submission))
