import pandas as pd
import argparse
from pathlib import Path

from src import dataconf
from src import nan_filter
from src import features_selection
from src import label_encoder


def explore_dataset(conf_file=None, df=None, edit=False):
    if (conf_file is None and df is None) or (conf_file is not None and df is not None):
        raise AttributeError("One and only one of conf_file and dataframe attirbute shall be defined.")
    if edit is True and conf_file is None:
        raise AttributeError("Can't edit dataframe conf file if no conf_file is provided...")
    if df is None:
        dataset = dataconf.DataConf()
        dataset.import_from_json(conf_file)
        df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep)
    for col in df.columns:
        unique = df[col].unique()
        if len(unique) <= 20:
            val = f'uniques : {unique}'
        else:
            val = f'min, max = [{df[col].min()} - {df[col].max()}]'
        print(f'df[{col}] = {list(df[col][:10].values)} | {val} | nan count = {df[col].isnull().sum()}')
    print(f'df shape = {df.shape}')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("file", type=str, help="Configuration file, json format")
    parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                       help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    args = parse.parse_args()
    dataset = dataconf.DataConf()
    try:
        dataset.import_from_json(args.file)
        df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep)
        df_clean = nan_filter.nan_filter(dataset=df,
                                         output=None,
                                         header=dataset.header,
                                         index_col=dataset.index_col,
                                         threshold=dataset.nan_column_threshold,
                                         force=args.force,
                                         verbosity=args.verbosity)
        if not args.force and input("Do you want to continue with feature selection (y/n) ? ") == "n":
            exit(0)
        df_clean, label_coder = label_encoder.label_encoder(input_df=df_clean,
                                                            ordinal=dataset.feature_type["ordinal"],
                                                            nominal=dataset.feature_type["nominal"],
                                                            exclude=None,
                                                            drop_first=True)
        df_clean = features_selection.features_selection(dataset=df_clean,
                                                         output=dataset.cleaned_dataset,
                                                         header=dataset.header,
                                                         index_col=dataset.index_col,
                                                         target_col=dataset.tcol,
                                                         threshold=dataset.cross_correlation_threshold,
                                                         useless_feature=dataset.feature_type["useless"],
                                                         verbose=args.verbosity,
                                                         force=args.force)
        explore_dataset(df=df_clean)
    except IOError as err:
        print(f"Error: {err}")
        exit(0)

