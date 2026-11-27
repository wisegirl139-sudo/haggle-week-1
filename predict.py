from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

here = Path(__file__).parent
train = pd.read_csv(here / "train.csv")
test = pd.read_csv(here / "test.csv")


def make_clues(df):
    # drop the id and answers, turn word columns into yes/no columns
    X = df.drop(columns=["project_id", "hours_spent", "log_hours", "hours_bucket"], errors="ignore")
    X = pd.get_dummies(X).astype(float)
    # new clues: one thing divided by another
    pairs = [("lines_added_total", "num_commits"), ("lines_added_total", "active_commit_days"),
             ("num_commits", "active_commit_days"), ("lines_of_code", "num_source_files"),
             ("lines_of_code", "num_commits"), ("lines_of_code", "active_commit_days"),
             ("lines_of_code", "lines_added_total"), ("active_commit_days", "project_span_days")]
    for a, b in pairs:
        X[a + "_per_" + b] = X[a] / (X[b] + 1)
    return X


X = make_clues(train)
# give the test table exactly the same columns as the training table
X_test = make_clues(test).reindex(columns=X.columns, fill_value=0)
y = np.log1p(train["hours_spent"])

models = [
    ExtraTreesRegressor(n_estimators=600, min_samples_leaf=3, max_features=0.5, random_state=0),
    GradientBoostingRegressor(learning_rate=0.02, n_estimators=800, max_depth=4, subsample=0.7,
                              max_features=0.5, min_samples_leaf=10, random_state=0),
    SVR(C=2, gamma=0.004, epsilon=0.2),
]
guesses = []
for m in models:
    pipe = make_pipeline(FunctionTransformer(np.log1p), SimpleImputer(strategy="median"),
                         StandardScaler(), m)
    pipe.fit(X, y)
    guesses.append(pipe.predict(X_test))

# CatBoost gets the original columns, with words left as words
drop = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
cat_train = train.drop(columns=drop)
cat_test = test.drop(columns=drop, errors="ignore")[cat_train.columns].copy()
words = list(cat_train.select_dtypes(exclude="number").columns)
cat_train[words] = cat_train[words].fillna("missing").astype(str)
cat_test[words] = cat_test[words].fillna("missing").astype(str)

cat = CatBoostRegressor(iterations=1600, learning_rate=0.03, depth=6, l2_leaf_reg=5,
                        cat_features=words, random_seed=42,
                        verbose=False, allow_writing_files=False)
cat.fit(cat_train, y)
guesses.append(cat.predict(cat_test))

# average the four models, then undo the log
hours = np.clip(np.expm1(np.mean(guesses, axis=0)), 0.1, 140.7)
out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": hours})
out.to_csv(here / "predictions.csv", index=False)
print(out.head())