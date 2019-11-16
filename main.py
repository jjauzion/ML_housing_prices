import pandas as pd
import argparse
from pathlib import Path

from src import dataconf
from src import nan_filter
from src import features_selection
from src import label_encoder


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file", type=str, help="Configuration file, json format")
    parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                       help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    args = parse.parse_args()
    data_conf = dataconf.DataConf()
    try:
        data_conf.import_from_json(args.file)
        nan_filter.nan_filter(data_conf.raw_dataset, data_conf.cleaned_dataset,
                              header=data_conf.header,
                              index_col=data_conf.index_col,
                              threshold=data_conf.nan_column_threshold,
                              force=args.force,
                              verbosity=args.verbosity)
        if not args.force and input("Do you want to continue with feature selection (y/n) ? ") == "n":
            exit(0)
        label_encoder.encode_cat_feature(data_conf.cleaned_dataset,
                                         output_dataset=data_conf.cleaned_dataset,
                                         header=data_conf.header,
                                         index_col=data_conf.index_col,
                                         ordinal_feature=data_conf.feature_type["ordinal"],
                                         nominal_feature=data_conf.feature_type["nominal"],
                                         exclude_feature=None)
        df = features_selection.features_selection(data_conf.cleaned_dataset,
                                                   output=data_conf.cleaned_dataset,
                                                   header=data_conf.header,
                                                   index_col=data_conf.index_col,
                                                   target_col=data_conf.tcol,
                                                   threshold=data_conf.cross_correlation_threshold,
                                                   useless_feature=data_conf.feature_type["useless"],
                                                   verbose=args.verbosity,
                                                   force=args.force)
        for col in df.columns:
            unique = df[col].unique()
            if len(unique) <= 20:
                val = f'uniques : {unique}'
            else:
                val = f'min, max = [{df[col].min()} - {df[col].max()}]'
            print(f'df[{col}] = {list(df[col][:10].values)} | {val}')
        print(f'df shape = {df.shape}')
    except IOError as err:
        print(f"Error: {err}")
        exit(0)

