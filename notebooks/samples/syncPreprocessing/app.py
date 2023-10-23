import json 
import os

def process(data:str, base_path:str): # O nome da função (process) é que deve ser passado no campo 'preprocess_reference'
    """
    Função usada para fazer o pré-processamento da base de dados
    Essa função deve estruturar os passos que o Neomaril executará para transformar a base de dados de entrada
    em uma nova base de dados operações de pré-processamento, sejam elas filtragens, extração de features, etc
    
    Parâmetros
    ---------
    data : str
        Os dados que serão usados pelo modelo que deverão chegar no formato string
    base_path : str
        O caminho para o arquivo do modelo e outros arquivos complementares

    Retorno
    -------
    dict | str no formato JSON:
        Um dicionário ou string no format de um JSON válido contendo as chaves e os valores correspondentes 
        aos dados, que formam a base de dados transformada.
    """

    # Carrega os dados do JSON para um objeto python
    data = json.loads(data)
    
    # Cria uma lista com os parâmetros da base de dados que devem ser mantidos
    parameters_to_keep = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
                       'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry',
                       'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error',
                       'area_error', 'smoothness_error', 'compactness_error', 'concavity_error',
                       'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius',
                       'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness',
                       'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

    # Cria dicionário que irá armazenar os valores escolhidos
    result = {}
    
    # Varre o objeto python a procura dos parâmetros que devem ser mantidos e adiciona no dicionário result
    for parameter in parameters_to_keep:
        result[parameter] = data.get(parameter, 0)
    
    # Retorna o dicionário com a nova base de dados
    return result