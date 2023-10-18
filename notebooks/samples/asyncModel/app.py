import os
from joblib import load
import pandas as pd


def score(data_path, model_path):# O nome da função (score) é que deve ser passado no campo 'model_reference'
    """
    Função usada para executar o modelo assíncrono com base no resultado do treino.
    Essa função deve estruturar os passos que o Neomaril executará para retornar o resultado
    da aplicação do modelo para os dados reais. 
    
    Na função o usuário pode usar variáveis de ambiente carregadas a partir de um arquivo .env,
    como exemplificado no código nas linhas 37-40.
    O Neomaril traz a opções de carregar duas variáveis de ambiente:
    modelFileName : str
        Que contém o nome do arquivo do modelo treinado (exemplo nas linhas 42-43)
    inputFileName : str
        Que contém o nome do arquivo da base de dados que foi feito upload (exemplo nas linhas 42-43)

    Parâmetros
    ---------
    data_path : str
        O caminho para os dados que serão usados pelo modelo que deverão chegar no formato string
    base_path : str
        O caminho para o arquivo do modelo e outros arquivos complementares

    Retorno
    -------
    str:
        O caminho com os arquivos resultantes da execução do modelo
    """

    ## Variáveis de ambiente carregadas de um arquivo fornecido pelo usuário no campo 'env'
    # my_var = os.getenv('MY_VAR')
    # if my_var is None:
    #    raise Exception("Could not find `env` variable")

    ## Variável de ambiente carregada do Neomaril com nome do arquivo do modelo (usado em alternativa a linha 40)
    # with open(base_path+os.getenv('modelFileName'), 'rb') as f:
    with open(model_path+"/model.pkl", 'rb') as f:
        model = load(f)

    ## Variável de ambiente carregada do Neomaril com nome da base de dados (usado em alternativa a linha 45)
    # X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))
    X = pd.read_csv(data_path+"/dados.csv")
    df = X.copy()

    df['proba'] = model.predict_proba(X)[:,1]
    df['pred'] = model.predict(X)

    output = data_path+'/output.csv'

    df.to_csv(output, index=False)

    return output