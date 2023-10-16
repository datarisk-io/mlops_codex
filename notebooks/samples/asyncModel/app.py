import os
from joblib import load
import pandas as pd


def score(data_path, model_path):
    
    model = load(model_path+"/model.pkl")

    X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))
    df = X.copy()

    df['proba'] = model.predict_proba(X)[:,1]
    df['pred'] = model.predict(X)

    output = data_path+'/output.csv'

    df.to_csv(output, index=False)

    return output