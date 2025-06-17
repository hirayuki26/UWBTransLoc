# Libraries
from time import time
# from os.path import exists, join
# from os import mkdir
from numpy import mean, std, sum, min, delete
from pandas import read_csv, concat

# Hyper-Pareameters / CONSTANTS
SRC_NULL = 100 # Original Null Value
DST_NULL = -100 # Changed Null Value
MIN_WAPS = 0 # Minimum number of WAPS per sample.

def load_data(fname, place_name, drop_columns=None, dst_null=DST_NULL):
    '''
    Loads data from CSV files.
    Splits the datasets into data and labels (X and Y).
    Replaces Null values and sets all lower null values to the replaced value.
    Normalizes data between 0 and 1 where 0 is weak intensity and 1 is strong intensity.

    Parameters: fname  : (str) file name of data - *.csv
                drop_columns : (list) column names to be removed from data
                dst_null     : (int) the value to change all null values to

    Returns   : x      : (Dataframe) features
                y      : (Dataframe) labels (coordinates)
    '''
    tic = time()  # Start function performance timer

    data = read_csv(f"./data/{place_name}/csv/" + fname + '.csv')

    Y = data[['x', 'y']]  # 'x' と 'y' 列をラベルとして抽出
    X = data.drop(columns=['x', 'y'])  # 'x' と 'y' 列を削除して特徴量として抽出

    if drop_columns:  # Drop useless columns if there are any specified.
        X.drop(columns=drop_columns, inplace=True)

    # Change null value to new value and set all lower values to it.
    X.replace(SRC_NULL, dst_null, inplace=True)
    X[X < dst_null] = dst_null

    # Normalize data between 0 and 1 where 1 is strong signal and 0 is null
    X /= min(X)
    X = 1 - X

    toc = time()  # Report function performance timer
    print("Data Load Timer: %.2f seconds" % (toc - tic))

    return X, Y