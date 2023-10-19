import pandas as pd
import shap
from joblib import load

# Função 1: o nome dela deve ser passado no campo 'preprocess_reference'
def parse(input_path, output_path):
    """
    Deve ser usada para transformar os dados brutos do modelo salvos no mesmo formato 
    que os dados do treino.
    
    Parâmetros
    ---------
    input_path : str
        O caminho dos dados para abrir os arquivos de variáveis
    output_path: str
        O caminho dos dados para abrir os arquivos de resultados do modelo.

    Retorno
    -------
    DataFrame tuple:
        Uma tupla de DataFrames, um com as variáveis e outro com os resultados do modelo,
        com as mesmas colunas que os dados de treino.
    """
    
    df_input = pd.read_csv(input_path)
    df_output = pd.read_csv(output_path)
    
    return (df_input, df_output)


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
