import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from . import utils
from . import label_encoder


def categorize_column(df, numeric_cat_column=None, inplace=False):
    """
    Convert all columns of dtype = "object" to numerical code. Also convert columns provided with numeric_cat_column
    :param df:                  [Pandas Dataframe] Dataframe with categorical columns to be converted.
    :param numeric_cat_column:  [list or None] list of numeric columns to be considered as label and converted to code.
    :param inplace:             [Bool] If True, modify the input Dataframe inplace
    :return:                    [Pandas Dataframe] dataframe with code instead of labels
    """
    if numeric_cat_column is None:
        numeric_cat_column = []
    df_code = df.copy(deep=True) if not inplace else df
    for col in df_code.select_dtypes("object"):
        df_code[col] = pd.Categorical(df_code[col])
        df_code[col] = df_code[col].cat.codes
    for col in numeric_cat_column:
        df_code[col] = pd.Categorical(df_code[col])
        df_code[col] = df_code[col].cat.codes
    return df_code


def _get_col2del(corr_feature, target_corr, useless_feature=None):
    """
    Create a list of feature that can be deleted from a correlated feature list. For each correlated feature,
    one feature will be added to the del_feature list only if the features are not already in the list.
    :param corr_feature:    [Pandas Dataframe] Multi-index dataframe of the correlated feature. Dataframe format:
                            Index1      Index2      Col
                            feature1    feature2    correlation_value
    :param target_corr:     [Pandas Dataframe] Feature correlation to the target. Dataframe format:
                            Index   Col
                            Feature correlation_value
    :param useless_feature: [list of str] name of features already identified as to be deleted.
    :return:                [list of str] name of features that can be deleted
    """

    del_feature = [] if useless_feature is None else copy.copy(useless_feature)
    for elm in corr_feature.index:
        if elm[0] not in del_feature and elm[1] not in del_feature:
            if target_corr[elm[0]] < target_corr[elm[1]]:
                del_feature.append(elm[0])
            else:
                del_feature.append(elm[1])
    return del_feature


def _plot_correlation(target_corr, feature_corr_list, feature_corr_matrix):
    fig1 = plt.figure()
    target_corr.sort_values().plot.bar()
    plt.title("Features correlation to the target")
    plt.xlabel("Features")
    plt.ylabel("Correlation coeff value")
    tmp = pd.DataFrame(feature_corr_list)
    tmp.plot.hist()
    plt.title("Features cross correlation coefficient")
    plt.xlabel("Correlation coeff value")
    top10_corr = feature_corr_list.sort_values(ascending=False)[:10]
    print("TOP 10 cross correlation:\n", top10_corr)
    top10_corr = top10_corr.unstack()
    top10_feature = set(top10_corr.index.to_list() + top10_corr.columns.to_list())
    fig2 = plt.figure()
    top10_corr_matrix = feature_corr_matrix.loc[top10_feature, top10_feature]
    sns.heatmap(top10_corr_matrix, square=True, linewidths=0.5, linecolor="Black",
                fmt=".1f", annot=True, cbar_kws={"shrink": 0.70}, vmax=1, center=0, vmin=-1, cmap="PiYG")
    plt.title("TOP 10 Cross correlation matrix")
    plt.show()


def correlated_features(df, target_col, output=None, index_col=None, header=0, threshold=0.9, verbose=1, force=False):
    """
    Interactive selection of correlated features.
    Create a list of feature that can be deleted based on the correlation value
    :param df:                  [Pandas DataFrame] input DataFrame
    :param target_col:          [str] target column name
    :param output:              [str] output file to save the cleaned dataframe as a csv
    :param index_col:           [int] Column of the dataset to use as the row labels of the DataFrame.
    :param header:              [int or None] Row number to use as the column names and start of the data.
    :param threshold:           [0 < float < 1] Correlation value above witch feature are considered correlated.
    :param force:               [Bool] If True, enforce "yes" answer to any prompt
    :param verbose:             [0 or 1 or 2] Verbosity level. Note that a verbosity of 0 will set force param to True
    :return:                    [(Pandas DataFrame, list)] (cleaned DataFrame, list of column names that can be deleted)
    """
    corr = df.corr()
    feature_corr_matrix = corr.drop(target_col)
    feature_corr_matrix = feature_corr_matrix.drop(target_col, axis=1)
    target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
    feature_corr_list = feature_corr_matrix.where(np.tril(feature_corr_matrix, k=-1).astype(np.bool)).stack()
    feature_corr_list = feature_corr_list.sort_values(ascending=False)
    if verbose > 1 and not force:
        print("Scatter plot of features best correlated to the target:")
        feature = list(target_corr.abs().sort_values(ascending=False)[:10].index)
        utils.scatter_multi(df, target_col, x_column=feature, y_title=target_col, x_title=feature)
        # for feature in target_corr.abs().sort_values(ascending=False)[:10].index:
            # utils.scatter(df, target_col, x_column=feature, y_title=target_col, x_title=feature)
        _plot_correlation(target_corr, feature_corr_list, feature_corr_matrix)
    correlated_feature = feature_corr_list[feature_corr_list > threshold]
    end = False if not force else True
    col2del = _get_col2del(correlated_feature, target_corr)
    while not end:
        if verbose > 0:
            print(f"Feature correlation to the target:\n{target_corr}")
            print(f"Correlated features above threshold (={threshold}):\n{correlated_feature}")
            print(f"List of correlated feature that can be deleted: {col2del}")
        print("\nDo you want to delete the feature selected above? This will create a new dataset, no data loss there!")
        end = utils.prompt_validation_or_new_threshold()
        if end == "exit":
            exit(0)
        if end != "yes" and end is not None:
            threshold = float(end)
            correlated_feature = feature_corr_list[feature_corr_list > threshold]
            col2del = _get_col2del(correlated_feature, target_corr)
            end = False
    df_out = df.drop(columns=col2del, errors="ignore")
    if output is not None:
        df_out.to_csv(output, sep=',', index=False if index_col is None else True,
                      header=False if header is None else True)
        if verbose > 0:
            print("Dataset after correlation cleaning saved to '{}'".format(output))
    return df_out, col2del
