import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def train_model(base_path):
    """
    Função usada para treinar o modelo com base em um conjunto de dados fornecido.
    Essa função deve estruturar os passos que o Neomaril terá que executar para retornar o
    conjunto de informações resultantes do treino. 

    Parâmetros
    ---------
    base_path : str
        O caminho de pastas para os arquivos que serão usados, bem como produzidos, no treino.
        Por exemplo: "/path/to/treino/customizado/experimento1"

    Retorno
    -------
    X_train: DataFrame
        Os dados que serão usados para treinar o modelo
    y_train: DataFrame
        Dataframe/array/série de target que será usado para treinar o modelo
    model_output: DataFrame
        Dataframe/array/série com os resultados do modelo treinado. 
        Que podem ser os valores/probabilidades previstos, classes ou qualquer outra informação útil. 
        Essa informação precisa estar na saída do modelo implantado futuro para ser usada no monitoramento.
    pipeline: Pipeline
        A instância do modelo ajustado final. 
        Idealmente, deve ser uma classe Pipeline do Scikit-Learn, mas qualquer outra classe de algoritmo que implemente o método get_params funciona. 
        Isso será salvo como model.pkl com cloudpickle <https://github.com/cloudpipe/cloudpickle> ou com o método save_model se a classe de algoritmo tiver isso.
    metrics: dict
        Um dicionário com cada chave como um métrica. 
        Você pode usar qualquer nome para a chave da métrica e salvar quantos quiser, mas o valor deve ser numérico. 
        Por exemplo: {“auc_train”: 0,7, “auc_test”: 0,65}
    extra: string list
        Uma lista opcional de nomes de arquivos para arquivos extras que são gerados no treinamento. 
        Que podem ser gráficos, conjuntos de validação, etc. Eles precisam ser salvos no mesmo caminho (base_path) que é fornecido como parâmetro da função.
    """

    # Adicionar uma env qualquer aqui O.o

    df = pd.read_csv(base_path+"/dados.csv")    # Carrega a base de dados, 'dados.csv' deve ser o nome do arquivo
    X = df.drop(columns=['target'])             # Separa a base de dados da coluna com os targets
    y = df[["target"]]                          # Salva os targets num DataFrame
    
    pipe = make_pipeline(SimpleImputer(), LGBMClassifier())     # Define quais serão os passos para treinar o modelo
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")  # Validação dos resultados usando a métrica 'auc'
    f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")   # Validação dos resultados usando a métrica 'f1'
    pipe.fit(X, y)  # Ajusta a base de dados ao target integrando-os ao Pipeline

    results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})  # Constrói o DataFrame com os resultados
    
    return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}

