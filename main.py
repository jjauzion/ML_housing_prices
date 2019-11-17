import pandas as pd
import argparse
import matplotlib.pyplot as plt

from src import dataconf
from src import nan_filter
from src import features_selection
from src import label_encoder
from src import explore


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file", type=str, help="Configuration file, json format")
    parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                       help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    parse.add_argument("-e", "--edit", action="store_true", help="Interactive edition of the configuration file")
    args = parse.parse_args()
    try:
        if args.edit:
            dataset = explore.explore_dataset(conf_file=args.file, edit=True)
        else:
            dataset = dataconf.DataConf()
            dataset.import_from_json(args.file)
        df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep,
                         na_values=dataset.nan_values)
        df_clean = nan_filter.nan_filter(df=df,
                                         output=None,
                                         header=dataset.header,
                                         index_col=dataset.index_col,
                                         threshold=dataset.nan_column_threshold,
                                         force=args.force,
                                         verbosity=args.verbosity,
                                         replace_nan=dataset.replace_nan)
        if not args.force and input("Do you want to continue with feature selection (y/n) ? ") == "n":
            exit(0)
        df_clean, label_coder = label_encoder.label_encoder(input_df=df_clean,
                                                            ordinal=dataset.feature_type["ordinal"],
                                                            nominal=dataset.feature_type["nominal"],
                                                            exclude=None,
                                                            drop_first=True,
                                                            custom_scale=dataset.ordinal_scale)
        df_clean = features_selection.features_selection(df=df_clean,
                                                         output=dataset.cleaned_dataset,
                                                         header=dataset.header,
                                                         index_col=dataset.index_col,
                                                         target_col=dataset.target_col,
                                                         threshold=dataset.cross_correlation_threshold,
                                                         useless_feature=dataset.feature_type["useless"],
                                                         verbose=args.verbosity,
                                                         force=args.force)
    except IOError as err:
        print(f"Error: {err}")
        exit(0)

