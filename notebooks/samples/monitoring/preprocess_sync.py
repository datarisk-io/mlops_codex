import pandas as pd
import numpy as np
import json
import shap
from cloudpickle import load

def parse(input_data, output_data):
    
    df = pd.DataFrame(data=json.loads(input_data), index=[0])
    df.loc[df['mean_perimeter'] < 150, 'mean_perimeter'] = None
    
    output = pd.DataFrame(data=json.loads(output_data), index=[0])
    
    return (df, output)


def get_shap(data, model_path):

    with open(model_path+"/model.pkl", 'rb') as f:
        model = load(f)

    explainer = shap.Explainer(model[-1])
    shap_values = explainer.shap_values(model[:-1].transform(data))[0]

    return pd.DataFrame(data=shap_values, columns=data.columns)
