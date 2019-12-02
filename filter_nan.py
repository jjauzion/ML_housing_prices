import pandas as pd

from src import nan_filter
from src import dataconf
from src import utils


def print_(string, verbosity):
    if verbosity > 0:
        print(string)


if __name__ == "__main__":
    args = utils.parse_main_args("file", "conf_output", "force", "verbosity")
    output_conf = args.conf_output if args.conf_output is not None else args.file
    dataset, df, _, _, _ = utils.import_df_from_dataconf(args.file, drop=True)
    print_("NaN filtering...".center(40, "-"), args.verbosity)
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
    dataset.to_json(output_conf, verbosity=args.verbosity)
    if args.verbosity > 0:
        print('\n--------------------------\nNaN filtering done!')
        print(f'Dataframe shape after NaN filtering: {df_clean.shape}')
        print(f'{len(deleted_col)} col deleted : {deleted_col}')
        print(f'{len(deleted_line)} line deleted : {deleted_line}')
        print(f'Data configuration file saved to "{output_conf}"')
