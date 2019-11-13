import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def features_selection(dataset, output, target_col, numeric_cat_column=None, header=0,
                       threshold=0.9, non_informative_col=None, verbose=1, force=False):
    """

    :param dataset:             [str] Dataset file, csv format expected.
    :param output:
    :param target_col:
    :param numeric_cat_column:
    :param header:              [int or None] Row number to use as the column names and start of the data.
    :param threshold:
    :param non_informative_col: [list]
    :param force:               [Bool] If True, enforce "yes" answer to any prompt
    :param verbose:             [0 or 1 or 2] Verbosity level. Note that a verbosity of 0 will set force param to True
    :return:
    """
    df = pd.read_csv(dataset, header=header)
    categorize_column(df, numeric_cat_column=numeric_cat_column, inplace=True)
    corr = df.corr()
    feature_corr = corr.drop(target_col)
    feature_corr = feature_corr.drop(target_col, axis=1)
    target_corr = corr[target_col].drop(target_col).sort_value(ascending=False)
    corr_list = feature_corr.where(np.tril(feature_corr, k=-1).astype(np.bool)).stack().sort_values(ascending=False)
    top_cor = corr_list[corr_list > threshold]
    end = False
    # get col2del
    while not end:
        # print
        # prompt for yes or new threshold
        # get col2del
        col2del = non_informative_col
        for elm in top_cor.index:
            if elm[0] not in col2del and elm[1] not in col2del:
                if target_corr[elm[0]] < target_corr[elm[1]]:
                    col2del.append(elm[0])
                else:
                    col2del.append(elm[1])
        if verbose > 0:
            print("Feature correlation to the target:\n", target_corr)
            print("Correlated features:\n", top_cor)
            print("List of column that should be deleted: {}".format(col2del))
        print("\nDo you want to delete the feature selected above? This will create a new dataset, no data loss there!")
        end = input("Type 'yes' to confirm OR 'exit' OR enter a new threshold value (val between 0 and 1).\n")
        if end == "exit":
            exit(0)
        elif end != "yes":
            try:
                threshold = float(end)
                if not (0 <= threshold <= 1):
                    raise ValueError(f"{end} is not a valid threshold. Threshold shall be between 0 and 1")
            except ValueError:
                print(f"{end} is not a valid threshold. Threshold shall be between 0 and 1")
                threshold = threshold
            else:
                top_cor = corr_list[corr_list > threshold]
    if verbose > 1:
        fig1 = plt.figure()
        target_corr.sort_values().plot.bar()
        plt.title("Features correlation to the target")
        plt.xlabel("Features")
        plt.ylabel("Correlation coeff value")
        tmp = pd.DataFrame(corr_list)
        tmp.plot.hist()
        plt.title("Features cross correlation coefficient")
        plt.xlabel("Correlation coeff value")
        fig2 = plt.figure()
        sns.heatmap(feature_corr, square=True, linewidths=0.5, linecolor="Black", fmt=".1f", annot=True, cbar_kws={"shrink":0.70}, vmax=1, center=0, vmin=-1, cmap="PiYG")
        plt.title("Cross correlation matrix")
        plt.show()
    df_out = df.drop(columns=col2del)
    df_out.to_csv(output, sep=',', index=False)
    if verbose > 0:
        print("cleaned data saved to '{}'".format(output))
