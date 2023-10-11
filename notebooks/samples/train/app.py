import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


def train_model(base_path):
    df = pd.read_csv(base_path+"/dados.csv")
    X = df.drop(columns=['target'])
    y = df[["target"]]
    
    pipe = make_pipeline(SimpleImputer(), LGBMClassifier())
    auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    f_score = cross_val_score(pipe, X, y, cv=5, scoring="f1")
    pipe.fit(X, y)

    results = pd.DataFrame({"pred": pipe.predict(X), "proba": pipe.predict_proba(X)[:,1]})
    
    return {"X_train": X, "y_train": y, "model_output": results, "pipeline": pipe, "metrics": {"auc": auc.mean(), "f1_score": f_score.mean()}}

