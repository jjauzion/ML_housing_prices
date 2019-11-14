import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy

from . import utils


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


def get_col2del(corr_feature, target_corr, useless_feature=None):
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


def plot_correlation(target_corr, feature_corr_list, feature_corr_matrix):
    fig1 = plt.figure()
    target_corr.sort_values().plot.bar()
    plt.title("Features correlation to the target")
    plt.xlabel("Features")
    plt.ylabel("Correlation coeff value")
    tmp = pd.DataFrame(feature_corr_list)
    tmp.plot.hist()
    plt.title("Features cross correlation coefficient")
    plt.xlabel("Correlation coeff value")
    fig2 = plt.figure()
    sns.heatmap(feature_corr_matrix, square=True, linewidths=0.5, linecolor="Black", fmt=".1f", annot=True,
                cbar_kws={"shrink": 0.70}, vmax=1, center=0, vmin=-1, cmap="PiYG")
    plt.title("Cross correlation matrix")
    plt.show()


def features_selection(dataset, output, target_col, numeric_cat_column=None, header=0,
                       threshold=0.9, useless_feature=None, verbose=1, force=False):
    """
    Interactive filtering of correlated features
    :param dataset:             [str] Dataset file, csv format expected.
    :param output:              [str] output file
    :param target_col:          [str] target column name
    :param numeric_cat_column:  [list of str] Name of the column that are categorical but with numeric values.
                                Those column cannot be automatically detected as categorical and hence shall be defined.
    :param header:              [int or None] Row number to use as the column names and start of the data.
    :param threshold:           [0 < float < 1] Correlation value above witch feature are considered correlated.
    :param useless_feature:     [list of str] Name of features identified as non informative for the prediction.
    :param force:               [Bool] If True, enforce "yes" answer to any prompt
    :param verbose:             [0 or 1 or 2] Verbosity level. Note that a verbosity of 0 will set force param to True
    :return:
    """
    df = pd.read_csv(dataset, header=header)
    categorize_column(df, numeric_cat_column=numeric_cat_column, inplace=True)
    corr = df.corr()
    feature_corr_matrix = corr.drop(target_col)
    feature_corr_matrix = feature_corr_matrix.drop(target_col, axis=1)
    target_corr = corr[target_col].drop(target_col).sort_values(ascending=False)
    feature_corr_list = feature_corr_matrix.where(np.tril(feature_corr_matrix, k=-1).astype(np.bool)).stack()
    feature_corr_list = feature_corr_list.sort_values(ascending=False)
    if verbose > 1 and not force:
        plot_correlation(target_corr, feature_corr_list, feature_corr_matrix)
    correlated_feature = feature_corr_list[feature_corr_list > threshold]
    end = False if not force else True
    col2del = get_col2del(correlated_feature, target_corr, useless_feature=useless_feature)
    while not end:
        if verbose > 0:
            print(f"Feature correlation to the target:\n{target_corr}")
            print(f"Correlated features above threshold (={threshold}):\n{correlated_feature}")
            print(f"List of column that should be deleted: {col2del}")
        print("\nDo you want to delete the feature selected above? This will create a new dataset, no data loss there!")
        end = utils.prompt_validation_or_new_threshold()
        if end == "exit":
            exit(0)
        if end != "yes" and end is not None:
            threshold = float(end)
            correlated_feature = feature_corr_list[feature_corr_list > threshold]
            col2del = get_col2del(correlated_feature, target_corr, useless_feature=useless_feature)
            end = False
    df_out = df.drop(columns=col2del)
    df_out.to_csv(output, sep=',', index=False)
    if verbose > 0:
        print("Dataset after correlation cleaning saved to '{}'".format(output))
