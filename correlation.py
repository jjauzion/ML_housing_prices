import pandas as pd

from src import features_selection
from src import dataconf
from src import utils
from src import error_lib


def print_(string, verbosity):
    if verbosity > 0:
        print(string)


if __name__ == "__main__":
    args = utils.parse_main_args("file", "conf_output", "force", "verbosity", "transform", "encode", "scale", "drop")
    output_conf = args.conf_output if args.conf_output is not None else args.file
    try:
        dataset, df, _, _, _ = utils.import_df_from_dataconf(args.file,
                                                             drop=args.drop,
                                                             encode=args.encode,
                                                             transform=args.transform,
                                                             scale=args.standardize)
    except error_lib.FileError as err:
        print(f"{err}")
        exit(0)
    print_("Features Correletion".center(40, "-"), args.verbosity)
    df_clean, deleted_col = features_selection.correlated_features(df=df,
                                                                   output=None,
                                                                   header=dataset.header,
                                                                   index_col=dataset.index_col,
                                                                   target_col=dataset.target_col,
                                                                   threshold=dataset.cross_correlation_threshold,
                                                                   verbose=args.verbosity,
                                                                   force=args.force)
    dataset.change_feature_type(deleted_col, new_type=dataset.useless)
    dataset.to_json(output_conf, verbosity=args.verbosity)
    if args.verbosity > 0:
        print('\n--------------------------\nNaN filtering done!')
        print(f'Dataframe shape after feature correlation filtering: {df_clean.shape}')
        print(f'{len(deleted_col)} col deleted : {deleted_col}')
        print(f'Data configuration file saved to "{output_conf}"')
