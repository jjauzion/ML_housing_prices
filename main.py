import pandas as pd
import argparse
import matplotlib.pyplot as plt

from src import dataconf
from src import nan_filter
from src import features_selection
from src import label_encoder
from src import explore
from src import preprocess_feature
from src import processing


def print_(string, verbosity):
    if verbosity > 0:
        print(string)


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
                         na_values=dataset.nan_values, keep_default_na=False)
        for feature, value in dataset.replace_nan.items():
            df[feature].fillna(value, inplace=True)
        print_("NaN filtering".center(40, "-"), args.verbosity)
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
        print_(f'Dataframe shape after NaN filtering: {df_clean.shape}', args.verbosity)
        print_("Transforming features".center(40, "-"), args.verbosity)
        trans_feature = preprocess_feature.transform_feature(df_clean, dataset.transform, delete_original=True)
        dataset.change_feature_type(trans_feature, new_type=dataset.useless)
        print_(f'New dataframe:\n{df_clean}', args.verbosity)
        print_(f'Dataframe shape after feature transformation: {df_clean.shape}', args.verbosity)
        print_("Features Correletion".center(40, "-"), args.verbosity)
        df_clean, deleted_col = features_selection.correlated_features(df=df_clean,
                                                                       output=None,
                                                                       header=dataset.header,
                                                                       index_col=dataset.index_col,
                                                                       target_col=dataset.target_col,
                                                                       threshold=dataset.cross_correlation_threshold,
                                                                       verbose=args.verbosity,
                                                                       force=args.force)
        dataset.change_feature_type(deleted_col, new_type=dataset.useless)
        print_(f'Dataframe shape after feature correlation filtering: {df_clean.shape}', args.verbosity)
        print_("Label Encoding".center(40, "-"), args.verbosity)
        df_clean.drop(columns=dataset.feature_type[dataset.useless], inplace=True, errors="ignore")
        df_clean, label_coder, binary_col = label_encoder.label_encoder(input_df=df_clean,
                                                                        ordinal=dataset.feature_type["ordinal"],
                                                                        nominal=dataset.feature_type["nominal"],
                                                                        exclude=None,
                                                                        drop_first=True,
                                                                        infer_nominal=False,
                                                                        custom_scale=dataset.ordinal_scale)
        print_(f'List of binary column added :    {binary_col}', args.verbosity)
        print_(f'List of encoded ordinal column : {dataset.feature_type["ordinal"]}', args.verbosity)
        print_("Features Correletion #2".center(40, "-"), args.verbosity)
        df_clean, deleted_col = features_selection.correlated_features(df=df_clean,
                                                                       output=None,
                                                                       header=dataset.header,
                                                                       index_col=dataset.index_col,
                                                                       target_col=dataset.target_col,
                                                                       threshold=dataset.cross_correlation_threshold,
                                                                       verbose=args.verbosity,
                                                                       force=args.force)
        dataset.change_feature_type(deleted_col, new_type=dataset.useless)
        binary_col = list(set(binary_col) - set(deleted_col))
        df_clean, scale = preprocess_feature.data_standardization(df_clean, except_col=binary_col)
        dataset.to_json(output_conf, verbosity=args.verbosity)
        df_clean.to_csv(dataset.cleaned_dataset, sep=',', index=dataset.index_col, header=dataset.header)
        print_(f'New dataframe:\n{df_clean}', args.verbosity)
        print_(f'Dataframe saved to "{dataset.cleaned_dataset}"', args.verbosity)
        X_train, y_train, X_test, y_true = preprocess_feature.split_dataset(df_clean, dataset.target_col, ratio=0.8)
        model, y_pred = processing.fit(X_train, y_train, X_test, y_name="LinearReg")
        print_(f'Result:\n{pd.concat((y_pred, y_true), axis=1).sort_values(by="LinearReg")}', args.verbosity)
        test_score = model.score(X_test, y_true)
        train_score = model.score(X_train, y_train)
        print_(f'Test score = {test_score} ; Train score = {train_score}', args.verbosity)
    except IOError as err:
        print(f'Error: {err}')
        exit(0)

