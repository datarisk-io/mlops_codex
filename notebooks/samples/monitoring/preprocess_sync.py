import pandas as pd
import numpy as np
import json
import shap
from joblib import load

# Função 1: o nome dela deve ser passado no campo 'preprocess_reference'
def parse(data):
    """
    Deve ser usada para transformar os dados brutos do modelo salvos no mesmo formato que os dados do treino.
    
    Parâmetros
    ---------
    data : str
        Os dados que serão usados pelo modelo que deverão chegar no formato string

    Retorno
    -------
    DataFrame:
        Dataframe com as mesmas colunas que os dados de treino
    """
    
    # Constrói um DataFrame com os dados de entrada
    df = pd.DataFrame(data=json.loads(data), index=[0])

    # Limpa a variável mean_perimeter se seu valor for inferior a 150
    df.loc[df['mean_perimeter'] < 150, 'mean_perimeter'] = None
    
    # Retorna oDataFrame modificado
    return df


# Função 2: o nome dela deve ser passado no campo 'shap_reference'
def get_shap(data, model_path):
    """
    Usada para calcular o SHAP.
    
    Parâmetros
    ---------
    data : str
        Os valores para cálculo do SHAP
    model_path: str
        O caminho para o arquivo do modelo

    Retorno
    -------
    DataFrame:
        Dataframe com os mesmos nomes de colunas que os dados do treino e os valores do SHAP
    """

    # Constrói o modelo a ser executado com base no arquivo de modelo
    with open(model_path+"/model.pkl", 'rb') as f:
        model = load(f)

    # Constrói o Explainer e calcula so valores SHAP para os dados do modelo
    explainer = shap.Explainer(model[-1])
    shap_values = explainer.shap_values(model[:-1].transform(data))

    # Retorna o resultado do SHAP para as colunas do treino
    return pd.DataFrame(data=shap_values, columns=data.columns)
