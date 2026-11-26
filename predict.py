#all imports
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from pathlib import Path
here = Path(__file__).parent    

teach = pd.read_csv(here / "train.csv")
test = pd.read_csv(here / "test.csv")
labels=["project_id", "hours_spent", "log_hours", "hours_bucket"] #our flags 
cols=teach.drop(columns=labels).select_dtypes(include=["number", "bool"]).columns
x=teach[cols].astype(float)
x_test=test[cols].astype(float)
y=np.log1p(teach["hours_spent"])