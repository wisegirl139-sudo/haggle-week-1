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
models = [
    ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=0),
    HistGradientBoostingRegressor(learning_rate=0.03, random_state=0),
    SVR(),
]
guesses=[]
for m in models :
    pipe=make_pipeline(FunctionTransformer(np.log1p),SimpleImputer(strategy="median"),StandardScaler(),m)
    pipe.fit(x,y)
    guesses.append(pipe.predict(x_test)) #just had a test on flipping append dequeu enque pop omds i cant see this rn 

hours=np.expm1(np.mean(guesses,axis=0))
hours= np.clip(hours,0.1,140.7)
out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": hours})
out.to_csv(here/"predictions.csv",index=False)
print(out.head())
    