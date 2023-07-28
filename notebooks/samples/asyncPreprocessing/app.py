import pandas as pd

def process(data_path):
    X = pd.read_csv(data_path+'/input.csv')
    df = X.copy()

    csv_filename = data_path + '/output.csv'

    df.to_csv(csv_filename, index=False)

    return csv_filename
