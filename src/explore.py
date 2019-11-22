from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src import dataconf


def explore_dataset(conf_file=None, dataframe=None, edit=False, save_file=None):
    plt.ion()
    plt.show()
    if (conf_file is None and dataframe is None) or (conf_file is not None and dataframe is not None):
        raise AttributeError("One and only one of conf_file and dataframe attirbute shall be defined.")
    if edit is True and conf_file is None:
        raise AttributeError("Can't edit dataframe conf file if no conf_file is provided...")
    if dataframe is None:
        dataset = dataconf.DataConf()
        dataset.import_from_json(conf_file)
        df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep,
                         na_values=dataset.nan_values)
    else:
        df = dataframe.copy(deep=True)
    feature_type = {
        "useless": [],
        "nominal": [],
        "ordinal": [],
        "binary": [],
        "count": [],
        "time": [],
        "continuous": []
    }
    for i, col in enumerate(df.columns):
        unique = df[col].unique()
        if len(unique) <= 20:
            val = f'uniques : {unique}'
        else:
            val = f'min, max = [{df[col].min()} - {df[col].max()}]'
        nan_count = df[col].isnull().sum()
        print(f'df[{col}] = {list(df[col][:10].values)} | {val} | nan count = {nan_count}')
        if edit is True:
            if df[col].dtype == "object":
                if nan_count > 0:
                    df[col].fillna("NaN", inplace=True)
                label = preprocessing.LabelEncoder()
                data = pd.DataFrame(data=label.fit_transform(df[col]), columns=[col])
                dtype = "nominal"
            else:
                data = pd.DataFrame(data=df[col], columns=[col])
                dtype = "continuous"
            fig = plt.figure(i)
            data.plot.hist()
            plt.title(col)
            plt.legend()
            plt.draw()
            plt.pause(0.001)
            plt.close(fig)
            ask = True
            while ask:
                ask = False
                ret = input(f'detected type : {dtype}. Press'
                            f' enter to validate or choose another : 0=useless, 1=ordinal, 2=nominal, 3=continuous\n')
                if ret == "exit":
                    exit(0)
                elif ret == "0":
                    dtype = "useless"
                elif ret == "1":
                    dtype = "ordinal"
                elif ret == "2":
                    dtype = "nominal"
                elif ret == "3":
                    dtype = "continuous"
                elif ret != "":
                    print(f'"{ret}" is not a valid value.')
                    ask = True
            feature_type[dtype].append(col)
    if edit is True:
        dataset.feature_type = feature_type
        if save_file is None:
            print("WARNING: no save file as been given. The new conf won't be saved to a file.")
            confirmation = input("Enter 'yes' to continue without save. Otherwise enter a valid file path")
            if confirmation != "yes":
                if Path(confirmation).is_file():
                    save_file = confirmation
                else:
                    save_file = Path(conf_file).parent / f'{Path(conf_file).stem}_new{Path(conf_file).suffix}'
                    print(f'"{confirmation}" is not a valid path file. New conf will be saved to "{save_file}"')
    if save_file is not None:
        dataset.to_json(save_file)
        print(f'Data conf saved to {save_file}')
    print(f'df shape = {df.shape}')
    return dataset
