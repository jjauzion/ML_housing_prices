import pandas as pd
import numpy as np
import sklearn.preprocessing as skl


def transform_feature(dataframe, transform_dic, delete_original=False):
    """
    Create new features from existing feature in the dataframe based on the transformation_dict dictionary.
    :param dataframe:
    :param transform_dic:
    :param delete_original:
    :return:
    """
    to_delete = []
    for feature, trans in transform_dic.items():
        if trans["operator"] == "+":
            dataframe[feature] = dataframe[trans["from"]].sum(axis=1)
        elif trans["operator"] == "-":
            dataframe[feature] = dataframe[trans["from"][0]]
            for feat in trans["from"][1:]:
                dataframe[feature] = dataframe[feature].sub(dataframe[feat])
        if "delete_original" not in trans or trans["delete_original"] == "True":
            to_delete += trans["from"]
    to_delete = list(set(to_delete))
    if delete_original:
        dataframe.drop(columns=to_delete, inplace=True)
    return to_delete


def data_standardization(df, columns="all", except_col=None):
    """
    Standardize the given dataframe on the specified columns
        z = (x - u) / s
    :param df:              [Pandas DataFrame] dataframe to standardize
    :param columns:         [list or "all"] list of column names to be standardize
    :param except_col:      [list] list of column names not to be standardize
    :return:                [tuple(pandas DataFrame, sklearn StandardScaler)] standardized dataframe and the scale used.
    """
    except_col = except_col if except_col is not None else []
    df_std = df.copy(deep=True) if columns == "all" else df.copy(deep=True)[columns]
    scale = skl.StandardScaler()
    df_std.drop(columns=except_col, inplace=True)
    array_std = scale.fit_transform(df_std)
    df_std = pd.concat((pd.DataFrame(data=array_std, index=df_std.index, columns=df_std.columns), df[except_col]), axis=1)
    return df_std, scale


def split_dataset(df, target_col, ratio=0.8):
    """
    Split a dataset in two
    :param df:              [pandas DatafFrame] dataframe to be splitted
    :param target_col:      [str] Name of the target column
    :param ratio:           [float] ratio for the split. If ratio = 0.5 both output dataframe will have the same size
    :return:                [tuple(dataframe, dataframe, dataframe, dataframe)] df_1_x, df_1_y, df_2_x, df_2_y
    """
    # df = pd.concat((pd.DataFrame(np.ones(shape=(df.shape[0], 1)), index=df.index, columns=["Bias"]), df), axis=1)
    # print(df)
    df_1 = df.sample(frac=ratio)
    df_2 = df.drop(df_1.index)
    return df_1.drop(columns=target_col), df_1[target_col], df_2.drop(columns=target_col), df_2[target_col]


if __name__ == "__main__":
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    split_dataset(df, "b")
    df = pd.DataFrame(data=np.random.randint(0, 10, (10, 3)), columns=["a", "b", "c"])
    print(f'\n{df}')
    xt, yt, xte, yte = split_dataset(df, "c")
    print(f'xtrain:\n{xt}\nytrain:\n{yt}\nxtest:\n{xte}\nytest:\n{yte}')

