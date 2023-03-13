import pandas as pd
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def train_model(base_path):
    df = pd.read_csv(base_path+"/dados.csv")
    X = df.drop(columns=['target'])
    y = df[["target"]]
    
    pipe = make_pipeline(SimpleImputer(), XGBClassifier())
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
    pipe.fit(X, y)

    results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
    results.proba.hist().get_figure().savefig(base_path+'/probas.png', format='png')
    
    return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, 'extras': [base_path+'/probas.png'],
            "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}
