#all imports
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from pathlib import Path
import pandas as pd

here = Path(__file__).parent
test = pd.read_csv(here / "test.csv")

out = pd.DataFrame({"project_id": test["project_id"], "hours_spent": 6})
out.to_csv(here / "predictions.csv", index=False)
print(out.head())