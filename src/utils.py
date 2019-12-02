import argparse
import pandas as pd

from . import dataconf
from . import preprocess_feature
from . import label_encoder
from . import nan_filter


def import_df_from_dataconf(file, drop=False, transform=False, encode=False, scale_data=False, verbosity=1):
    label_coder = None
    scale_x = None
    scale_y = None
    binary_col = None
    dataset = dataconf.DataConf()
    dataset.import_from_json(file)
    dataset.delete_duplicate()  # in case of bad input from user
    df = pd.read_csv(dataset.raw_dataset, header=dataset.header, index_col=dataset.index_col, sep=dataset.sep,
                     na_values=dataset.nan_values, keep_default_na=False)
    if transform:
        trans_feature = preprocess_feature.transform_feature(df, dataset.transform, delete_original=True)
        dataset.change_feature_type(trans_feature, new_type=dataset.useless)
    if drop:
        df.drop(columns=dataset.feature_type[dataset.useless], inplace=True, errors="ignore")
        df.drop(dataset.useless_line, inplace=True)
    if encode:
        df = nan_filter.replace_nan_with(df, dataset.replace_nan, verbosity=1)
        df, label_coder, binary_col = label_encoder.label_encoder(input_df=df,
                                                                  ordinal=dataset.feature_type["ordinal"],
                                                                  nominal=dataset.feature_type["nominal"],
                                                                  exclude=None,
                                                                  drop_first=True,
                                                                  infer_nominal=False,
                                                                  custom_scale=dataset.ordinal_scale)
    if scale_data:
        df, scale_x, scale_y = preprocess_feature.data_standardization(df, target_col=dataset.target_col,
                                                                       except_col=binary_col)
    return dataset, df, label_coder, scale_x, scale_y


def prompt_validation_or_new_threshold(message=None):
    """
    Prompt user for "yes" or "exit" or a float value between 0 and 1.

    :param message: [str] Message to be printed to the user.
    :return:        [None] If user input is none of the above, return None.
                    [str] If user input is "yes" or "exit", return "yes" or "exit".
                    [float] If user input is a valid float between 0 and 1, return the value as a float.
    """
    if message is None:
        message = "Type 'yes' to confirm OR 'exit' OR enter a new threshold value (val between 0 and 1).\n"
    end = input(message)
    if end == "exit" or end == "yes":
        return end
    try:
        threshold = float(end)
        if not (0 <= threshold <= 1):
            raise ValueError(f"{threshold} is not a valid threshold. Threshold shall be between 0 and 1")
    except ValueError:
        print(f"{end} is not a valid threshold. Threshold shall be between 0 and 1")
        return None
    return threshold


def parse_main_args(*args):
    parse = argparse.ArgumentParser()
    if "file" in args:
        parse.add_argument("file", type=str, help="Configuration file, json format")
    if "conf_output" in args:
        parse.add_argument("-conf_output", type=str, default=None,
                           help="path where to save the config file. If None (default), input file is overwritten")
    if "force" in args:
        parse.add_argument("-f", "--force", action="store_true", help="Force 'yes' answer to any prompt")
    if "verbosity" in args:
        parse.add_argument("-v", "--verbosity", choices=[0, 1, 2], default=1, type=int,
                           help="Verbosity level. 0: silent ; 1: printed output ; 2: print + plot")
    if "edit" in args:
        parse.add_argument("-e", "--edit", action="store_true", help="Interactive edition of the configuration file")
    if "transform" in args:
        parse.add_argument("-t", "--transform", action="store_true",
                           help="Transform the dataframe feature according to the dataconf file")
    if "scale" in args:
        parse.add_argument("-s", "--standardize", action="store_true", help="Standardize the dataframe feature")
    if "encode" in args:
        parse.add_argument("-c", "--encode", action="store_true", help="Encode the nominal and ordinal feature")
    if "drop" in args:
        parse.add_argument("-d", "--drop", action="store_true",
                           help="Drop features and/or observations according to the dataconf file")
    return parse.parse_args()


