import pandas as pd
import numpy as np
import json
import shap
from joblib import load

# Função 1: o nome dela deve ser passado no campo 'preprocess_reference'
def parse(data:str):
    """
    Deve ser usada para transformar os dados brutos do modelo salvos no mesmo formato que os dados do treino.
    
    Parâmetros
    ---------
    data : str
        Os dados que serão usados pelo modelo que deverão chegar no formato string

    Retorno
    -------
    Pandas DataFrame:
        Dataframe com as mesmas colunas que os dados de treino
    """
    
    # Constrói um DataFrame com os dados de entrada. Os dados chegam como um JSON string então temos que tranforma-los em um dicionário
    df = pd.DataFrame(data=json.loads(data), index=[0])

    # Aqui podemos fazer o mesmo preprocessamento que fizemos no treino.
    # Nesse exemplo limpamos a variável mean_perimeter se seu valor for inferior a 150
    df.loc[df['mean_perimeter'] < 150, 'mean_perimeter'] = None
    
    # Retorna o DataFrame modificado
    return df


# Função 2: o nome dela deve ser passado no campo 'shap_reference'
def get_shap(data:pd.DataFrame, model_path:str):
    """
    Usada para calcular o SHAP.
    
    Parâmetros
    ---------
    data : Pandas DataFrame
        Os valores para cálculo do SHAP. 
        O dataframe enviado já vai ter passado pela função 1, então já vai ter todas as variaveis usadas no treino
    model_path: str
        O caminho para o arquivo do modelo

    Retorno
    -------
    Pandas DataFrame:
        Dataframe com os mesmos nomes de colunas que os dados do treino e os valores do SHAP
    """

    # Carrega o modelo já treinado para ser executado com base no arquivo de modelo passado como parâmetro
    with open(model_path+"/model.pkl", 'rb') as f:
        model = load(f)

    # Constrói o Explainer e calcula so valores SHAP para os dados do modelo
    explainer = shap.Explainer(model[-1])
    shap_values = explainer.shap_values(model[:-1].transform(data))

    # Retorna o resultado do SHAP para as colunas do treino. 
    # Ele deve ter a mesma quantidade de linhas e colunas que o dataframe que foi enviado para a função, e ter o mesmo nome das colunas
    return pd.DataFrame(data=shap_values, columns=data.columns)
