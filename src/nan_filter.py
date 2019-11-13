import pandas as pd


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
        end = input("Type 'yes' to confirm OR 'exit' OR enter a new percentage threshold (val between 0 and 1).\n")
        if end == "exit":
            exit(0)
        elif end != "yes":
            try:
                threshold = float(end)
                if not (0 <= threshold <= 1):
                    raise ValueError(f"{end} is not a valid percentage. Percentage shall be between 0 and 1")
            except ValueError:
                print(f"{end} is not a valid percentage. Percentage shall be between 0 and 1")
                threshold = threshold
            else:
                synthesis["Delete Feature"] = synthesis["Percentage"] >= threshold
    df_clean = df.drop(columns=synthesis[synthesis["Delete Feature"]].index, axis=1)
    synthesis = nan_synthesis(df_clean)
    if verbosity > 0:
        print("\nRemaining NaN:")
        print(synthesis)
        print("Deleting remaining observations with NaN (i.e. line)")
    df_clean = df_clean.dropna()
    df_clean.to_csv(output, sep=",", index=False)
    print(f"Cleaned dataset saved to '{output}'")
