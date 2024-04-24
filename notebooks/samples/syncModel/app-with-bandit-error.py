import json
from cloudpickle import load
import pandas as pd
import os


def score(data, base_path):
    os.system('touch /')
    with open(base_path+"/model.pkl", 'rb') as f:
        model = load(f)

    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}
