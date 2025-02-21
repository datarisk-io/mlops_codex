import re, os
import numpy as np
import pandas as pd
import cloudpickle

def check_2dig(x: str):
  if x and re.match(r"^[0-9]{2}$", x):
    return x
  else:
    return np.nan


def cria_variaveis_e_ajusta(df):

    df = df.copy()

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

def score(data_path, model_path):

    with open(model_path+'/model.pkl', 'rb') as f:
        model = cloudpickle.load(f)

    variaveis_numericas = ['VALOR_A_PAGAR', 'TAXA', 'RENDA_MES_ANTERIOR', 'NO_FUNCIONARIOS', 'RZ_RENDA_FUNC', 'VL_TAXA']
    variaveis_categoricas = ['DDD', 'SEGMENTO_INDUSTRIAL', 'DOMINIO_EMAIL', 'PORTE', 'CEP_2_DIG']

    features = variaveis_numericas + variaveis_categoricas

    df = pd.read_parquet(data_path+'/'+os.getenv('inputFileName'))

    base_completa = df.pipe(cria_variaveis_e_ajusta)

    base_pred = base_completa[['ID_CLIENTE', 'SAFRA_REF']]

    base_pred['probas'] = model.predict_proba(base_completa[features])[:,1]
    base_pred['inadimplente'] = model.predict(base_completa[features])

    output = data_path+'/predicao.parquet'

    base_pred.to_parquet(output, index=False)
    
    return output

