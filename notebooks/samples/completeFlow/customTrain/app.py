import re, os
import numpy as np
import pandas as pd

from feature_engine.encoding import OneHotEncoder
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def check_2dig(x: str):
  if x and re.match(r"^[0-9]{2}$", x):
    return x
  else:
    return np.nan


def cria_variaveis_e_ajusta(df):

    df = df.copy()

    df['DATA_PAGAMENTO'] = pd.to_datetime(df.DATA_PAGAMENTO)
    df['DATA_VENCIMENTO'] = pd.to_datetime(df.DATA_VENCIMENTO)

    # Criação de Variáveis
    df['RZ_RENDA_FUNC'] = df[['RENDA_MES_ANTERIOR','NO_FUNCIONARIOS']].apply(lambda x: x[0]/x[1] if x[1]> 0 else np.nan, axis=1)
    df['VL_TAXA'] = df[['TAXA','VALOR_A_PAGAR']].apply(lambda x: (x[0]/100)*x[1] if x[1]> 0 else np.nan, axis=1)

    # Ajustes de Variáveis
    df['DDD'] = df['DDD'].astype(str).apply(check_2dig)
    df['SEGMENTO_INDUSTRIAL'] = df['SEGMENTO_INDUSTRIAL'].astype(str).map({'Serviços':'SERVICOS','Comércio':'COMERCIO','Indústria':'INDUSTRIA'})
    df['DOMINIO_EMAIL'] = df['DOMINIO_EMAIL'].astype(str).map({'YAHOO':'YAHOO','HOTMAIL':'HOTMAIL','OUTLOOK':'OUTLOOK','GMAIL':'GMAIL','BOL':'BOL','AOL':'AOL'})
    df['PORTE'] = df['PORTE'].astype(str).map({'PEQUENO':'PEQUENO','MEDIO':'MEDIO','GRANDE':'GRANDE'})
    df['CEP_2_DIG'] = df['CEP_2_DIG'].astype(str).apply(check_2dig)

    # Preenchimento de nulos nas variáveis categóricas
    variaveis_categoricas =  ['DDD', 'SEGMENTO_INDUSTRIAL', 'DOMINIO_EMAIL', 'PORTE', 'CEP_2_DIG']

    df[variaveis_categoricas] = df[variaveis_categoricas].fillna('MISSING')
    return df

def train_model(df, base_path):
    base_completa = (
        # Cruzamento das bases
        df
        # Remove as amostras que são pessoas físicas
        .query('FLAG_PF.isna()')
        # Cria e ajusta as variáveis
        .pipe(cria_variaveis_e_ajusta)
        # Cria a variável target
        .assign(
            DIAS_DE_ATRASO = lambda df_: (df_['DATA_PAGAMENTO'] - df_['DATA_VENCIMENTO']).dt.days,
            FLAG_MAU = lambda df_: (df_['DIAS_DE_ATRASO']>=5).astype(int)
        )
    )

    variaveis_numericas = ['VALOR_A_PAGAR', 'TAXA', 'RENDA_MES_ANTERIOR', 'NO_FUNCIONARIOS', 'RZ_RENDA_FUNC', 'VL_TAXA']
    variaveis_categoricas = ['DDD', 'SEGMENTO_INDUSTRIAL', 'DOMINIO_EMAIL', 'PORTE', 'CEP_2_DIG']

    features = variaveis_numericas + variaveis_categoricas

    df_train = base_completa[base_completa['SAFRA_REF']<'2021-03-01'].copy()
    df_oot = base_completa[base_completa['SAFRA_REF']>='2021-03-01'].copy()

    df_train['SAFRA_REF'] = df_train['SAFRA_REF'].astype(str)

    model = Pipeline(steps = [
        ('Imputação de Valores Faltantes em Variáveis Numéricas', MeanMedianImputer(variables=variaveis_numericas, imputation_method='mean')),
        ('Normalizando Variáveis Numéricas', SklearnTransformerWrapper(MinMaxScaler(), variables=variaveis_numericas)),
        ('Imputação de Valores Faltantes em Variáveis Categóricas', CategoricalImputer(variables=variaveis_categoricas, imputation_method='missing')),
        ('Criando Variáveis Categóricas Dummies', OneHotEncoder(variables=variaveis_categoricas, top_categories=5)),
        ('Modelo de Regressão Logística', LogisticRegression(class_weight='balanced', max_iter=2))
    ])

    cross_validation_strategy = StratifiedKFold(n_splits=5)

    results = cross_validate(
        estimator=model,
        X=df_train[features],
        y=df_train['FLAG_MAU'],
        scoring=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'],
        cv=cross_validation_strategy
    )

    model.fit(df_train[features], df_train['FLAG_MAU'])


    train_probas = model.predict_proba(df_train[features])[:,1]

    model_outputs = df_train[['SAFRA_REF']]
    model_outputs['probas'] = train_probas

    mean_results = pd.DataFrame(results).mean().to_frame().T.to_dict(orient='records')[0]

    mean_results['roc_auc_oot'] = roc_auc_score(df_oot['FLAG_MAU'], model.predict_proba(df_oot[features])[:,1])

    roc_auc_by_safra = df_oot.groupby('SAFRA_REF').apply(lambda df_: roc_auc_score(df_['FLAG_MAU'], model.predict_proba(df_[features])[:,1]))

    roc_auc_by_safra.plot.bar().get_figure().savefig(base_path+'/auc_safra.png')

    importances = model[-1].coef_[0]
    names = model[-1].feature_names_in_
    
    varimp = pd.Series(importances, index=names).sort_values().tail(10)
    varimp.plot.barh().get_figure().savefig(base_path+'/varimp.png')
    
    return {"X_train": df_train[features+['SAFRA_REF']], "y_train": df_train[['FLAG_MAU']], "model_output": model_outputs, 
            "pipeline": model, "metrics": mean_results, 'extras': [base_path+'/auc_safra.png', base_path+'/varimp.png']}
