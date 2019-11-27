import pandas as pd
import numpy as np
from sklearn import preprocessing


class StandardDFScaler(preprocessing.StandardScaler):

    def fit_transform(self, X, y=None, **fit_params):
        """
        fit transform a pandas dataframe and return a pandas dataframe
        """
        std_df = super().fit_transform(X, y, **fit_params)
        return pd.DataFrame(data=std_df, index=X.index, columns=X.columns)

    def transform(self, X, copy=None):
        """
        transform a pandas dataframe and return a pandas dataframe
        """
        std_df = super().transform(X, copy)
        return pd.DataFrame(data=std_df, index=X.index, columns=X.columns)

    def inverse_transform(self, X, copy=None):
        """
        inverse transform a pandas dataframe and return a pandas dataframe
        """
        inv_df = super().inverse_transform(X, copy)
        return pd.DataFrame(data=inv_df, index=X.index, columns=X.columns)


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


def data_standardization(df, target_col, columns="all", except_col=None):
    """
    Standardize the given dataframe on the specified columns
        z = (x - u) / s
    :param df:              [Pandas DataFrame] dataframe to standardize
    :param columns:         [list or "all"] list of column names to be standardize
    :param except_col:      [list] list of column names not to be standardize
    :return:                [tuple(pandas DataFrame, sklearn StandardScaler)] standardized dataframe and the scale used.
    """
    except_col = except_col if except_col is not None else []
    df_tmp = df.copy(deep=True) if columns == "all" else df.copy(deep=True)[columns]
    df_tmp.drop(columns=except_col, inplace=True)
    df_y = df_tmp.loc[:, [target_col]]
    df_x = df_tmp.drop(columns=target_col)
    scaleX = StandardDFScaler()
    scaleY = StandardDFScaler()
    x_std = scaleX.fit_transform(df_x)
    y_std = scaleY.fit_transform(df_y)
    # x_std = pd.DataFrame(data=scaleX.fit_transform(df_x), index=df_x.index, columns=df_x.columns)
    # y_std = pd.DataFrame(data=scaleY.fit_transform(df_y), index=df_y.index, columns=df_y.columns)
    df_std = pd.concat((y_std, x_std, df.loc[:, except_col]), axis=1)
    return df_std, scaleX, scaleY


def split_dataset(df, target_col, ratio=0.8, seed=None):
    """
    Split a dataset in two
    :param df:              [pandas DatafFrame] dataframe to be splitted
    :param target_col:      [str] Name of the target column
    :param ratio:           [float] ratio for the split. If ratio = 0.5 both output dataframe will have the same size
    :param seed:            [int] seed value for the random splitting of the data set
    :return:                [tuple(dataframe, dataframe, dataframe, dataframe)] df_1_x, df_1_y, df_2_x, df_2_y
    """
    # df = pd.concat((pd.DataFrame(np.ones(shape=(df.shape[0], 1)), index=df.index, columns=["Bias"]), df), axis=1)
    # print(df)
    df_1 = df.sample(frac=ratio, random_state=seed)
    df_2 = df.drop(df_1.index)
    return df_1.drop(columns=target_col), df_1.loc[:, [target_col]], df_2.drop(columns=target_col), df_2.loc[:, [target_col]]


if __name__ == "__main__":
    df = pd.DataFrame({"a": [0, 1, 2, 3], "b": [10, 11, 12, 13]})
    split_dataset(df, "b")
    df = pd.DataFrame(data=np.random.randint(0, 10, (10, 3)), columns=["a", "b", "c"])
    print(f'\n{df}')
    xt, yt, xte, yte = split_dataset(df, "c")
    print(f'xtrain:\n{xt}\nytrain:\n{yt}\nxtest:\n{xte}\nytest:\n{yte}')

