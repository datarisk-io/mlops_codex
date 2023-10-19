import json
from joblib import load
import pandas as pd
import os


def score(data, base_path): # O nome da função (score) é que deve ser passado no campo 'model_reference'
    """
    Função usada para executar o modelo síncrono com base no resultado do treino.
    Essa função deve estruturar os passos que o Neomaril executará para retornar o resultado
    da aplicação do modelo para os dados reais. 
    
    Na função o usuário pode usar variáveis de ambiente carregadas a partir de um arquivo .env,
    como exemplificado no código nas linhas 39-42.
    Caso não queira deixar o nome do arquivo com as variáveis do modelo fixo, o Neomaril carrega 
    o nome desse arquivo na variável de ambiente (exemplo nas linhas 44-45):
    modelFileName : str
        Que contém o nome do arquivo do modelo treinado

    Parâmetros
    ---------
    data : str
        Os dados que serão usados pelo modelo que deverão chegar no formato string
    base_path : str
        O caminho para o arquivo do modelo e outros arquivos complementares

    Retorno
    -------
    dict | list dict| str no formato JSON:
        Um dicionário, lista de dicionários ou string no format de um JSON válido contendo 
        as chaves e os valores para os resultados de saída do modelo.
        Neste exemplo, usamos a chaves:
            pred: int
                Valor de predição do modelo
            proba: int 
                Probabilidade da pertencer a classe 'a'
    """

    ## Variáveis de ambiente carregadas de um arquivo fornecido pelo usuário no campo 'env'
    # my_var = os.getenv('MY_VAR')
    # if my_var is None:
    #    raise Exception("Could not find `env` variable")

    ## Variável de ambiente do Neomaril com nome do arquivo do modelo (usado em alternativa a linha 48)
    # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

    # Usado para construir o modelo a ser executado com base no arquivo de modelo passado como parâmetro
    with open(base_path+"/model.pkl", 'rb') as f:
        model = load(f)

    # Constrói o DataFrame com os dados de entrada
    df = pd.DataFrame(data=json.loads(data), index=[0])
    
    # Retorna os resultados da execução do modelo como um dicionário
    return {"pred": int(model.predict(df)), "proba": float(model.predict_proba(df)[0,1])}
