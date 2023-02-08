import pandas as pd
import numpy as np
import json
import shap
from joblib import load

def parse(data):
    
    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    return df


def get_shap(data, model_path):

    model = load(model_path+"/model.pkl")

    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(data)

    return pd.DataFrame(data=shap_values, columns=data.columns)
