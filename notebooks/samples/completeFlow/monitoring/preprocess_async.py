import re
import shap
import numpy as np
import pandas as pd
import cloudpickle

variaveis_numericas = ['VALOR_A_PAGAR', 'TAXA', 'RENDA_MES_ANTERIOR', 'NO_FUNCIONARIOS', 'RZ_RENDA_FUNC', 'VL_TAXA']
variaveis_categoricas = ['DDD', 'SEGMENTO_INDUSTRIAL', 'DOMINIO_EMAIL', 'PORTE', 'CEP_2_DIG']

features = variaveis_numericas + variaveis_categoricas


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

def build_df(input_path, output_path):
    df = pd.read_parquet(input_path)
    base_completa = (
        # Cruzamento das bases
        df
        # Remove as amostras que são pessoas físicas
        .query('FLAG_PF.isna()')
        # Cria e ajusta as variáveis
        .pipe(cria_variaveis_e_ajusta)
    )

    df['SAFRA_REF'] = df['SAFRA_REF'].astype(str)

    base_pred = pd.read_parquet(output_path)
    return base_completa[features+['SAFRA_REF']], base_pred

def get_shap(data, model_path):
  with open(model_path+'/model.pkl', 'rb') as f:
    model = cloudpickle.load(f)

  df_transf = model[:-1].transform(data[features])
  explainer = shap.LinearExplainer(model[-1], df_transf)
  shap_values = explainer.shap_values(df_transf)

  df_shap = pd.DataFrame(data=shap_values, columns=df_transf.columns)

  ddd_cols = [c for c in df_transf.columns if 'DDD' in c]

  df_shap['DDD'] = df_shap[ddd_cols].mean(axis=1)

  segmento_cols = [c for c in df_transf.columns if 'SEGMENTO_INDUSTRIAL' in c]

  df_shap['SEGMENTO_INDUSTRIAL'] = df_shap[segmento_cols].mean(axis=1)

  dominio_cols = [c for c in df_transf.columns if 'DOMINIO_EMAIL' in c]

  df_shap['DOMINIO_EMAIL'] = df_shap[dominio_cols].mean(axis=1)

  porte_cols = [c for c in df_transf.columns if 'PORTE' in c]

  df_shap['PORTE'] = df_shap[porte_cols].mean(axis=1)

  cep_cols = [c for c in df_transf.columns if 'CEP_2_DIG' in c]

  df_shap['CEP_2_DIG'] = df_shap[cep_cols].mean(axis=1)

  df_shap = df_shap.drop(columns=ddd_cols+segmento_cols+dominio_cols+porte_cols+cep_cols)

  return df_shap

