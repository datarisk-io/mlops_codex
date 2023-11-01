:orphan:

AutoML configuration
====================


{
    "period": "week", <string, Mandatory> indicates monitoring frequency running. Can be `day`, `week`, `month` or `year`
    "train_data" : {
        "NeomarilTrainingExecution": "1", <string, Optional> é o id de execução do treino (SOMENTE se foi treinado dentro do neomaril)
        "train_date_col": (optional[string]) o nome da coluna de data (de tempo mesmo) dos dados (NECESSÁRIO se não tiver o campo "train_date_ref")
        "train_date_ref": "2022-09-01", (optional[string]) a data de aquisição dos dados (NECESSÁRIO se não tiver o campo "train_date_col", ou seja é necessário conter um, e exclusivamente um, dos campos: "train_date_col" ou exclusivo "train_date_ref")
    }
    "input_cols": ["mean_radius", "mean_texture"] (list<string>) nome das colunas de features que o monitoramento executará. Precisa ser igual aos dados de treino no MLFlow e à saída da função de pré-processamento
    "output_cols": ["proba", "pred"](list<string>) nome das colunas de saída da execução do monitoramento. Precisa ser igual aos dados de saída do MLFlow e à saída do da predição do modelo.
}