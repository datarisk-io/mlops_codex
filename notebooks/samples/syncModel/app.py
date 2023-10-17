import json
from joblib import load
import pandas as pd


def score(data, base_path):
    """
    Função usada para executar o modelo para os dados de entrada, 
    retornando um dicionário, uma lista de dicionário ou uma string JSON 
    contendo a predição e a probabilidade quando aplicada aquele conjunto de dados

    Parâmetros
    ---------
    data : str
        Os dados em JSON que vão chegar pela requisição no formato string
    base_path : str
        O caminho para o arquivo do modelo e outros arquivos complementares

    Retorno
    -------
    dict { "pred": int, "proba": int }
        Um dicionário contendo a predição "pred" e a probabilidade "proba" para o conjunto de dados de entrada
    """
    with open(base_path+"/model.pkl", 'rb') as f:
        model = load(f)

    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}
