import pandas as pd

def process(data_path:str):  # O nome da função (process) é que deve ser passado no campo 'preprocess_reference'
    """
    Função usada para fazer o pré-processamento da base de dados
    Essa função deve estruturar os passos que o Neomaril executará para transformar a base de dados de entrada
    em uma nova base de dados operações de pré-processamento, sejam elas filtragens, extração de features, etc
    
    Caso não queira deixar o nome da base de dados fixo, o Neomaril carrega o nome desse arquivo
    na variável de ambiente (exemplo nas linhas 25-26):
    inputFileName : str
        Que contém o nome do arquivo da base de dados que foi feito upload

    Parâmetros
    ---------
    data : str
        Os dados que serão usados pelo modelo que deverão chegar no formato string

    Retorno
    -------
    str:
        O caminho com o(s) arquivo(s) com a base de dados transformada
    """

    ## Variável de ambiente carregada do Neomaril com nome da base de dados (usado em alternativa a linha 51)
    # X = pd.read_csv(data_path+'/'+os.getenv('inputFileName'))
    
    # Carrega os dados da base de entrada do arquivo para um DataFrame
    df = pd.read_csv(data_path+'/input.csv')

    # Cria uma lista com os parâmetros da base de dados que devem ser mantidos
    parameters_to_keep = [
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
        'mean_smoothness', 'mean_compactness', 'mean_concavity', 'mean_concave_points',
        'mean_symmetry', 'mean_fractal_dimension', 'radius_error', 'texture_error',
        'perimeter_error', 'area_error', 'smoothness_error', 'compactness_error',
        'concavity_error', 'concave_points_error', 'symmetry_error', 'fractal_dimension_error',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry',
        'worst_fractal_dimension'
    ]

    #Cria um DataFrame com os parâmetros que devem ser mantidos
    final_df = df[parameters_to_keep]

    # Cria o caminho com o nome do arquivo de output, nesse caso 'output.csv' 
    csv_filename = data_path + '/output.csv'

    # Transforma o DataFrame, com a predição e a probabilidade, para csv colocando no arquivo do caminho do output
    final_df.to_csv(csv_filename, index=False)

    # Retorna o caminho com o arquivo com os resultados da execução do modelo
    return csv_filename
