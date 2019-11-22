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
    parse.add_argument("-conf_output", type=str, default=None, help="path where to save the config file."
                                                                    "If None (default), input file is overwritten")
    parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                       help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    parse.add_argument("-e", "--edit", action="store_true", help="Interactive edition of the configuration file")
    args = parse.parse_args()
    output_conf = args.conf_output if args.conf_output is not None else args.file
    try:
        if args.edit:
            dataset = explore.explore_dataset(conf_file=args.file, edit=True, save_file=output_conf)
        else:
            dataset = dataconf.DataConf()
            dataset.import_from_json(args.file)
        dataset.delete_duplicate()  # in case of bad input from user
        df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep,
                         na_values=dataset.nan_values)
        df_clean, deleted_col, deleted_line = nan_filter.nan_filter(df=df,
                                                                    output=None,
                                                                    header=dataset.header,
                                                                    index_col=dataset.index_col,
                                                                    threshold=dataset.nan_column_threshold,
                                                                    force=args.force,
                                                                    verbosity=args.verbosity,
                                                                    nan_substitute=dataset.replace_nan)
        dataset.change_feature_type(deleted_col, new_type=dataset.useless)
        dataset.add_useless_line(deleted_line)
        if not args.force and input("Do you want to continue with feature selection (y/n) ? ") == "n":
            exit(0)
        df_clean, deleted_col = features_selection.correlated_features(df=df_clean,
                                                                       output=None,
                                                                       header=dataset.header,
                                                                       index_col=dataset.index_col,
                                                                       target_col=dataset.target_col,
                                                                       threshold=dataset.cross_correlation_threshold,
                                                                       verbose=args.verbosity,
                                                                       force=args.force)
        dataset.change_feature_type(deleted_col, new_type=dataset.useless)
        df_clean, label_coder = label_encoder.label_encoder(input_df=df_clean,
                                                            ordinal=dataset.feature_type["ordinal"],
                                                            nominal=dataset.feature_type["nominal"],
                                                            exclude=None,
                                                            drop_first=True,
                                                            custom_scale=dataset.ordinal_scale)
        dataset.to_json(output_conf, verbosity=args.verbosity)
        df_clean.to_csv(dataset.cleaned_dataset, sep=',', index=dataset.index_col, header=dataset.header)
        if args.verbosity > 0:
            print(f'New dataframe:\n{df_clean}')
            print(f'Dataframe saved to "{dataset.cleaned_dataset}"')
    except IOError as err:
        print(f'Error: {err}')
        exit(0)

