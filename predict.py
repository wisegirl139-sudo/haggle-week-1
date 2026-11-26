#all imports
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

teach=pd.read_csv("train.csv")
test=pd.read_csv(sys.argv[1] if len(sys.argv)>1 else "test.csv")
