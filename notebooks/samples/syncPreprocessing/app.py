import json 

def process(data, base_path): # O nome da função (process) é que deve ser passado no campo 'script_reference'
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

    data = json.loads(data)
    
    parameters_to_keep = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
                       'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry',
                       'mean_fractal_dimension', 'radius_error', 'texture_error', 'perimeter_error',
                       'area_error', 'smoothness_error', 'compactness_error', 'concavity_error',
                       'concave_points_error', 'symmetry_error', 'fractal_dimension_error', 'worst_radius',
                       'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness', 'worst_compactness',
                       'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

    result = {}
    
    for parameter in parameters_to_keep:
        result[parameter] = data.get(parameter, 0)
        
    return result