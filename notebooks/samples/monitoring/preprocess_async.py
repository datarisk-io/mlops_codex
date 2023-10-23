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
    Pandas DataFrame tuple:
        Uma tupla de DataFrames, um com as variáveis e outro com os resultados do modelo,
        com as mesmas colunas que os dados de treino.
    """
    
    # Nesse caso nós enviamos o caminho completo do arquivo. Então não precisa se preocupar com o nome do arquivo.
    df_input = pd.read_csv(input_path)
    df_output = pd.read_csv(output_path)

    # Aqui podemos fazer o mesmo preprocessamento que fizemos no treino.
    # Nesse exemplo limpamos a variável mean_perimeter se seu valor for inferior a 150
    df_input.loc[df_input['mean_perimeter'] < 150, 'mean_perimeter'] = None
    
    return (df_input, df_output)


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
