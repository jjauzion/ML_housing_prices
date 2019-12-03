import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import special

from . import dataconf
from . import preprocess_feature
from . import label_encoder
from . import nan_filter


def import_df_from_dataconf(file, drop=False, transform=False, encode=False, scale=False, log=False, unskew=False):
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
    if unskew:
        lam = 0.15
        for feat in dataset.unskew:
            df[feat] = special.boxcox1p(df[feat], lam)
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
    if log:
        for feature in dataset.log_transform:
            df[feature] = np.log1p(df[feature])
    if scale:
        df, scale_x, scale_y = preprocess_feature.data_standardization(df, target_col=dataset.target_col,
                                                                       except_col=binary_col)
    return dataset, df, label_coder, scale_x, scale_y


def prompt_validation_or_new_threshold(message=None, min_=0., max_=1.):
    """
    Prompt user for "yes" or "exit" or a float value between 0 and 1.

    :param message: [str] Message to be printed to the user.
    :param min_:     [float] Minimum valid value for the threshold. User is prompt another entry if entered val < min
    :param max_:     [float] Maximum valid value for the threshold. User is prompt another entry if entered val > max
    :return:        [None] If user input is none of the above, return None.
                    [str] If user input is "yes" or "exit", return "yes" or "exit".
                    [float] If user input is a valid float between 0 and 1, return the value as a float.
    """
    if message is None:
        message = f'Type "yes" to confirm OR "exit" OR enter a new threshold value (val between {min_} and {max_}).\n'
    end = input(message)
    if end == "exit" or end == "yes":
        return end
    try:
        threshold = float(end)
        if not (min_ <= threshold <= max_):
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
    if "unskew" in args:
        parse.add_argument("-u", "--unskew", action="store_true", help="Correct skew feature to a normal distribution")
    return parse.parse_args()


def scatter_multi(df, y_column, x_column=None, y_title="y", x_title="x", color_col=None):
    """
    create a scattter plot
    :param df:          [Pandas dataframe]
    :param y_column:    [str] y col of the dataframe to be used for y value
    :param x_column:    [list of str] x col of the dataframe to be used for x value. If None, will x be a np.arange(len(nb of sample))
    :param x_title:     [list of str] x axis title
    :param y_title:     [str] y axis title
    :param color_col:   [str] column to be used for color
    """
    def onpick3(event):
        ind = event.ind
        print(f'Select point index is : {df.iloc[ind].index.values}')
    if color_col is not None:
        color_scale = df[color_col].unique()
        colors = np.linspace(0, 1, len(color_scale))
        color_dict = dict(zip(color_scale, colors))
        color_func = np.vectorize(lambda val: color_dict[val])
        color_df = color_func(df[color_col])
    else:
        color_df = None
    fig = plt.figure()
    n = math.ceil(math.sqrt(len(x_column)))
    for i, feature in enumerate(x_column):
        plt.subplot(n, n, i + 1)
        plt.scatter(x=df[feature], y=df[y_column], c=color_df, picker=True)
        fig.canvas.mpl_connect('pick_event', onpick3)
        plt.xlabel(x_title[i])
        plt.ylabel(y_title)
    plt.show()


def scatter(df, y_column, x_column=None, y_title="y", x_title="x", color_col=None):
    """
    create a scattter plot
    :param df:          [Pandas dataframe]
    :param y_column:    [str] y col of the dataframe to be used for y value
    :param x_column:    [str] x col of the dataframe to be used for x value. If None, will x be a np.arange(len(nb of sample))
    :param x_title:     [str] x axis title
    :param y_title:     [str] y axis title
    :param color_col:   [str] column to be used for color
    """
    def onpick3(event):
        ind = event.ind
        print(f'Select point index is : {df.iloc[ind].index.values}')
    x = np.arange(df.shape[0]) if x_column is None else df.loc[:, x_column]
    if color_col is not None:
        color_scale = df[color_col].unique()
        colors = np.linspace(0, 1, len(color_scale))
        color_dict = dict(zip(color_scale, colors))
        color_func = np.vectorize(lambda val: color_dict[val])
        color_df = color_func(df[color_col])
    else:
        color_df = None
    fig = plt.figure()
    plt.scatter(x=x, y=df[y_column], c=color_df, picker=True)
    fig.canvas.mpl_connect('pick_event', onpick3)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


