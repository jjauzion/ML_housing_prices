import pandas as pd

from . import utils


def nan_synthesis(df):
    null = df.isnull().sum()
    percentage = null / df.isnull().count()
    synthesis = pd.concat((null, percentage), axis=1,
                          keys=("Total", "Percentage")).sort_values(ascending=False, by="Total")
    synthesis = synthesis[synthesis["Total"] != 0]
    return synthesis


def replace_nan_with(df, feature_dict):
    df_clean = df.copy(deep=True)
    for feature in feature_dict:
        df_clean[feature] = df_clean[feature].fillna(feature_dict[feature])
    return df_clean


def nan_filter(df, output=None, header=0, index_col=None, threshold=0.005, replace_nan=None,
               force=False, verbosity=1):
    """
    Interactive filter of NaN value in a dataframe by deletion of columns and / or rows.
    :param df:          [pandas DataFrame] DataFrame to clean
    :param output:      [str] output file
    :param header:      [int or None] Row number to use as the column names and start of the data.
    :param index_col:   [int] Column of the dataset to use as the row labels of the DataFrame.
    :param threshold:   [float] Percentage of NaN required for column deletion. Can be modified at run time.
    :param replace_nan: [dict] For each feature in the replace_nan dictionary, NaN is replaced with the provided value.
                        Ex: {"feature_1": 0} will replace NaN in column "feature_1" by 0.
    :param force:       [Bool] If True 'yes' answer is enforced at every user prompt request.
    :param verbosity:   [0 or 1 or 2] Verbosity level. Note that verbosity = 0 enforce yes answer
                        as does the force parameter.
    :return:            [Pandas Dataframe] Cleaned dataframe
    """
    if replace_nan is not None:
        df_clean = replace_nan_with(df, replace_nan)
    synthesis = nan_synthesis(df_clean)
    end = "no" if not force and verbosity > 0 else "yes"
    synthesis["Delete Feature"] = synthesis["Percentage"] >= threshold
    while end != "yes":
        print(synthesis)
        print("\nDo you want to delete the feature selected above? This will create a new dataset, no data loss there!")
        end = utils.prompt_validation_or_new_threshold()
        if end == "exit":
            exit(0)
        if end != "yes" and end is not None:
            threshold = float(end)
            synthesis["Delete Feature"] = synthesis["Percentage"] >= threshold
            end = False
    df_clean = df_clean.drop(columns=synthesis[synthesis["Delete Feature"]].index, axis=1)
    synthesis = nan_synthesis(df_clean)
    if verbosity > 0 and not force:
        print("\nRemaining NaN:")
        print(synthesis)
        print("Deleting remaining observations with NaN (i.e. line)")
    df_clean = df_clean.dropna()
    if output is not None:
        df_clean.to_csv(output, sep=",", index=False if index_col is None else True,
                        header=True if header is not None else False)
        print(f"Dataset after NaN filtering saved to '{output}'")
    return df_clean
