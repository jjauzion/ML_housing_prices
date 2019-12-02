import pandas as pd

from . import utils
from . import dataconf


def nan_synthesis(df):
    null = df.isnull().sum()
    percentage = null / df.isnull().count()
    synthesis = pd.concat((null, percentage), axis=1,
                          keys=("Total", "Percentage")).sort_values(ascending=False, by="Total")
    synthesis = synthesis[synthesis["Total"] != 0]
    return synthesis


def replace_nan_with(df, nan_substitute, verbosity=1):
    df_clean = df.copy(deep=True)
    for feature in nan_substitute:
        if feature not in df.columns:
            if verbosity > 0:
                print(f'WARNING: Can\'t apply NaN substitue for "{feature}": feature not found in the dataframe')
        else:
            df_clean[feature] = df_clean[feature].fillna(nan_substitute[feature])
    return df_clean


def nan_filter(df, output=None, header=0, index_col=None, threshold=0.005, nan_substitute=None,
               force=False, verbosity=1):
    """
    Interactive filter of NaN value in a dataframe by deletion of columns and / or rows.
    :param df:          [pandas DataFrame] DataFrame to clean
    :param output:      [str] output file where the cleaned dataframe will be saved as a csv file
    :param header:      [int or None] Row number to use as the column names and start of the data.
    :param index_col:   [int] Column of the dataset to use as the row labels of the DataFrame.
    :param threshold:   [float] Percentage of NaN required for column deletion. Can be modified at run time.
    :param nan_substitute:  [dict] For each feature in the nan_substitute, NaN is replaced with the provided value.
                            Ex: {"feature_1": 0} will replace NaN in column "feature_1" by 0.
    :param force:       [Bool] If True 'yes' answer is enforced at every user prompt request.
    :param verbosity:   [0 or 1 or 2] Verbosity level. Note that verbosity = 0 enforce yes answer
                        as does the force parameter.
    :return:            [(Pandas Dataframe, list, list)] Tuple as follow:
                        (Cleaned dataframe, list of feature name to be deleted, list of rows to be deleted)
    """
    df_clean = replace_nan_with(df, nan_substitute) if nan_substitute is not None else df.copy(deep=True)
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
    feature_to_delete = synthesis[synthesis["Delete Feature"]].index.to_list()
    df_clean = df_clean.drop(columns=feature_to_delete)
    synthesis = nan_synthesis(df_clean)
    if verbosity > 0 and not force:
        print("\nRemaining NaN:")
        print(synthesis)
        print("Deleting remaining observations with NaN (i.e. line)")
    line_to_delete = df_clean.index[df_clean.isnull().any(axis=1)].to_list()
    df_clean = df_clean.drop(line_to_delete)
    if output is not None:
        df_clean.to_csv(output, sep=",", index=False if index_col is None else True,
                        header=True if header is not None else False)
        print(f"Dataset after NaN filtering saved to '{output}'")
    return df_clean, feature_to_delete, line_to_delete
