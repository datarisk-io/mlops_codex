from joblib import load
import pandas as pd
from neomaril_codex.logging import Logger


def score(data_path, model_path):
    logger = Logger('Async')

    logger.debug("USING LOGGER")
    
    model = load(model_path+"/model.pkl")

    df = pd.read_csv(data_path+'/input.csv')

    if len(df) < 5:
        logger.warning("DF is less than 5 lines")

    df['score'] = 1000 * (1-model.predict_proba(df)[:,1])

    output = data_path+'/output.csv'

    df.to_csv(output, index=False)

    return output
