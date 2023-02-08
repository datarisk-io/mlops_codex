import json
from joblib import load
import pandas as pd


def score(data, base_path):
    model = load(base_path+"/model.pkl")

    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    return {"score": 1000 * (1-float(model.predict_proba(df)[0,1]))}
