import pandas as pd
import argparse
from pathlib import Path

from src import toolbox
from src import nan_filter
from src import features_selection


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file", type=str, help="Configuration file, json format")
    parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                       help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    args = parse.parse_args()
    data_conf = toolbox.DataConf()
    try:
        data_conf.import_from_json(args.file)
        nan_filter.nan_filter(data_conf.raw_dataset, data_conf.cleaned_dataset,
                              threshold=data_conf.nan_column_threshold, force=args.force, verbosity=args.verbosity)
        if not args.force and input("Do you want to continue with feature selection (y/n) ? ") == "n":
            exit(0)
        print("\n")
        features_selection.features_selection(data_conf.cleaned_dataset,
                                              output=data_conf.cleaned_dataset,
                                              numeric_cat_column=data_conf.numeric_cat_column,
                                              header=0,
                                              target_col=data_conf.tcol,
                                              threshold=data_conf.cross_correlation_threshold,
                                              non_informative_col=data_conf.feature_type["useless"],
                                              verbose=args.verbosity)
    except IOError as err:
        print(f"Error: {err}")
        exit(0)

