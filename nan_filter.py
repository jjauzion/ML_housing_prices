import pandas as pd
import argparse
from pathlib import Path

from src import toolbox
from src import check_arg


def nan_synthesis(df):
    null = df.isnull().sum()
    percentage = null / df.isnull().count()
    synthesis = pd.concat((null, percentage), axis=1,
                          keys=("Total", "Percentage")).sort_values(ascending=False, by="Total")
    synthesis = synthesis[synthesis["Total"] != 0]
    return synthesis


def nan_filter(input_file, output_file, threshold=0.5, force=False):
    data_conf = toolbox.DataConf()
    data_conf.import_from_json(args.file)
    df = pd.read_csv(data_conf.raw_data_file)
    synthesis = nan_synthesis(df)
    threshold = args.threshold / 100
    end = "no"
    while end != "yes":
        synthesis["Delete Feature"] = synthesis["Percentage"] >= threshold
        print(synthesis)
        print("\nDo you want to delete the feature selected above ? (This will create a new dataset, no data loss there!)")
        if not args.force:
            end = input("Type 'yes' to confirm OR 'exit' OR enter a new percentage threshold (val between 0 and 1).\n")
        else:
            end = "yes"
        if end == "exit":
            exit(0)
        elif end != "yes":
            try:
                threshold = float(end)
            except ValueError:
                print(f"'{end}' is not a valid value")
            if not (0 <= threshold <= 1):
                print(f"'{threshold}' is not a valid percentage. Please enter a value between 0 and 1")
                threshold = args.threshold
    df_clean = df.drop(columns=synthesis[synthesis["Delete Feature"]].index, axis=1)
    synthesis = nan_synthesis(df_clean)
    print("\nRemaining NaN:")
    print(synthesis)
    print("Delete remaining sample with NaN (ie line)")
    df_clean = df_clean.dropna()
    df_clean.to_csv(args.output, sep=",", index=False)
    print(f"Cleaned dataset saved to '{args.output}'")
    t = 1 / 0


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file", help="Configuration file, json format")
    parse.add_argument("-t", "--threshold", default=0.5, type=check_arg.is_positive,
                       help="Percentage of nan above which the feature should be deleted. Value must be >= 0 and <= 100.")
    parse.add_argument("-o", "--output", default="data/dataset_cleaned.csv", type=str,
                       help="Output file for the cleaned dataset")
    parse.add_argument("-yf", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    args = parse.parse_args()
    try:
        nan_filter(args.file, args.output, threshold=args.threshold, force=args.force)
    except IOError as err:
        print(f"Error: {err}")
        # print(f"Error: can't read json configuration file '{Path(args.file)}' because: {err.__class__.__name__}:{err}")
        exit(0)

