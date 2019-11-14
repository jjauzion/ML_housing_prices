import pandas as pd

from . import utils


def nan_synthesis(df):
    null = df.isnull().sum()
    percentage = null / df.isnull().count()
    synthesis = pd.concat((null, percentage), axis=1,
                          keys=("Total", "Percentage")).sort_values(ascending=False, by="Total")
    synthesis = synthesis[synthesis["Total"] != 0]
    return synthesis


def nan_filter(dataset, output, header=0, threshold=0.005, force=False, verbosity=1):
    """
    Interactive filter of NaN value in a dataframe by deletion of columns and / or rows.
    :param dataset:     [str] dataset file, csv format expected.
    :param output:      [str] output file
    :param header:      [int or None] Row number to use as the column names and start of the data.
    :param threshold:   [float] Percentage of NaN required for column deletion. Can be modified at run time.
    :param force:       [Bool] If True 'yes' answer is enforced at every user prompt request.
    :param verbosity:   [0 or 1 or 2] Verbosity level. Note that verbosity = 0 enforce yes answer
                        as does the force parameter.
    """
    df = pd.read_csv(dataset, header=header)
    synthesis = nan_synthesis(df)
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
    df_clean = df.drop(columns=synthesis[synthesis["Delete Feature"]].index, axis=1)
    synthesis = nan_synthesis(df_clean)
    if verbosity > 0 and not force:
        print("\nRemaining NaN:")
        print(synthesis)
        print("Deleting remaining observations with NaN (i.e. line)")
    df_clean = df_clean.dropna()
    df_clean.to_csv(output, sep=",", index=False)
    print(f"Dataset after NaN filtering saved to '{output}'")
