from joblib import load
import pandas as pd


def score(data_path, model_path):
    model = load(model_path+"/model.pkl")

    df = pd.read_csv(data_path+'/input.csv')

    df['score'] = 1000 * (1-model.predict_proba(df)[:,1])

    output = data_path+'/output.csv'

    df.to_csv(output, index=False)

    return output
