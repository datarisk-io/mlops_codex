:orphan:

AutoML configuration
====================


{
    "train_data":{
       "file_type": "csv", <string, Mandatory> Uploaded dataset file type. Can be `csv` or `parquet`
       "file_name": "dados.csv", <string, Mandatory> Uploaded dataset file name
       "sep": "," <string, Mandatory for CSV> Separator for the csv file. 
    },
    "model_flow":"classification", <string, Mandatory> Model class, for now can be only `classification`
    "target":"TARGET", <string, Mandatory> Name of the target column in the uploaded dataset
    "cat_cols":["ID"], <string, Optional> Name of the columns that need to be encoded as categorical. Default is a empty list (we will try to find categorical columns)
    "iterations":10, <int, Optional> How many pipelines combinations we will test. Default is 1.
    "metric":"ks", <string, Optional> Metric we will use to find the best model. For classification the options are `auc`, `precision`, `recall`, `f1`, `gini`, `ks`. Default is `auc`.
    "split_type":"random", <string, Optional> How we will split the training, validation and test datasets. Options are `random`, `stratified` (random but trying to get the same proportion of data between splits) and `oot` (validation is random, but test is split by date) .Default value is `random`
    "val_size": 0.2, <float, Optional> Proportion of the validation dataset to the full dataset. Default is 0.2
    "holdout_size": 0.1, <float, Optional> Proportion of the test dataset to the full dataset. Only used when `split_type` is `random` or `stratified`. Default is 0.1
    "stratify_col": "TARGET", <string, Optional> Which column to use to stratify the split (keeping the same proportion between splits). Only used when `split_type` is  `stratified`. Default is the target column
    "date_col": "DATE", <string, Optional> Which column to use to find the most recent records. Only used when `split_type` is  `oot`
    "oot_split_size": 0.1, <float, Optional> Fraction of the most recent data to use as test dataset. When `split_type` is  `oot` this or `split_date` must be informed. Default is 0.2
    "split_date": "2020-01-01", <string, Optional> Date to filter the test dataset. When `split_type` is  `oot` this or `oot_split_size` must be informed.
    "stages":{
       "models":["lightgbm"] <list[string], Optional> Algorithms to test. Options are `logeg`, `catboost`, `xgboost`, `lightgbm`, `rf`, `dt`. Default is use all
       "missing":["mean"] <list[string], Optional> Missing imputation methods to test. Options are `mean`, `median`, `tail` (replacing missing data by a value at left tail of the distribution), `random` and `none` (will only work if the algorithm alreay handle missing data). Will only be used if data has missing values. Default is use all
       "cleaner":["iqr"] <list[string], Optional> Outlier remover methods to test. Options are `iqr`, `rare` and `none`. Default is use all
       "encoding":["catboost"] <list[string], Optional> Categorical encoder methods to test. Options are `rankcount`, `catboost`, `count` and `dt`. Will only be used if data has categorical columns. Default is use all
       "preprocess":["none"] <list[string], Optional> Scaler methods to test. Options are `norm`, `robust`, `minmax`, `binarizer` and `none`. Default is use all
       "unbalance":["none"] <list[string], Optional> Target balacing methods to test. Options are `smote`, `random_under` and `none`. Will only be used if the minority class for the target is less than 10% of the data. Default is use all
    }
 }

