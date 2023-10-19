import os
from joblib import load
import pandas as pd


def score(data_path, model_path):# O nome da função (score) é que deve ser passado no campo 'model_reference'
    """
    Função usada para executar o modelo assíncrono com base no resultado do treino.
    Essa função deve estruturar os passos que o Neomaril executará para retornar o resultado
    da aplicação do modelo para os dados reais. 
    
    Na função o usuário pode usar variáveis de ambiente carregadas a partir de um arquivo .env,
    como exemplificado no código nas linhas 35-38.
    O Neomaril traz a opções de carregar duas variáveis de ambiente:
    modelFileName : str
        Que contém o nome do arquivo do modelo treinado (exemplo nas linhas 40-41)
    inputFileName : str
        Que contém o nome do arquivo da base de dados que foi feito upload (exemplo nas linhas 47-48)

    Parâmetros
    ---------
    data_path : str
        O caminho para os dados que serão usados pelo modelo que deverão chegar no formato string
    base_path : str
        O caminho para o arquivo do modelo e outros arquivos complementares

    Retorno
    -------
    str:
        O caminho com o(s) arquivo(s) com os resultados da execução do modelo.
        Neste exemplo, retornamos a tabela com os inputs e a predição e a probabilidade para cada 
        linha da tabela.
    """

    ## Variáveis de ambiente carregadas de um arquivo fornecido pelo usuário no campo 'env'
    # my_var = os.getenv('MY_VAR')
    # if my_var is None:
    #    raise Exception("Could not find `env` variable")

    ## Variável de ambiente carregada do Neomaril com nome do arquivo do modelo (usado em alternativa a linha 44)
    # with open(base_path+os.getenv('modelFileName'), 'rb') as f:

    # Usado para construir o modelo a ser executado com base no arquivo de modelo passado como parâmetro
    with open(model_path+"/model.pkl", 'rb') as f:
        model = load(f)

    ## Variável de ambiente carregada do Neomaril com nome da base de dados (usado em alternativa a linha 51)
    # X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))

    # Carrega os dados da base de entrada do arquivo para um DataFrame
    X = pd.read_csv(data_path+"/dados.csv")

    df = X.copy()   # Cria uma cópia do DataFrame com os dados de entrada

    df['proba'] = model.predict_proba(X)[:,1]   # Calcula a predição de cada entrada da tabela de dados
    df['pred'] = model.predict(X)               # Calcula a probabilidade de cada entrada da tabela de dados

    # Cria o caminho com o nome do arquivo de output, nesse caso 'output.csv' 
    output = data_path+'/output.csv'

    # Transforma o DataFrame, com a predição e a probabilidade, para csv colocando no arquivo do caminho do output
    df.to_csv(output, index=False)

    # Retorna o caminho com o arquivo com os resultados da execução do modelo
    return output