import sklearn.preprocessing as preprocessing
import pandas as pd
from pathlib import Path
import pickle
import collections
import numpy as np


class LabelEncoder:

    def __init__(self, classes=None):
        self._class = []
        if classes is not None:
            self.fit(classes)

    def fit(self, classes, order="keep"):
        """

        :param classes:
        :param order: 'keep' -> keep ordered of the classes list and assign number in this order
                      'sort' -> sort classes in ascending order first and then assign number in this order
        """
        if order == "sort":
            self._class = sorted(list(set(classes)))
        elif order == "keep":
            self._class = list(collections.OrderedDict.fromkeys(classes))

    def transform(self, classes, ignore_unknown_class=False):
        try:
            label = [float(self._class.index(elm)) for elm in classes]
        except TypeError:
            if isinstance(classes, bytes):
                classes = classes.decode('utf-8')
            label = float(self._class.index(classes))
        except ValueError:
            if ignore_unknown_class:
                print("WARNING: One of the given name is not a valid class name.\nGot '{}' ; Valid class name : '{}'"
                      .format(classes, self._class))
                return np.nan
            else:
                print("WARNING: One of the given name is not a valid class name.\nGot '{}' ; Valid class name : '{}'"
                      .format(classes, self._class))
                raise ValueError("One of the given name is not a valid class name.\nGot '{}' ; Valid class name : '{}'"
                                 .format(classes, self._class))
        return label

    def fit_transform(self, classes):
        self.fit(classes)
        return self.transform(classes)

    def inverse_transform(self, label):
        try:
            classes = [self._class[int(index)] for index in label]
        except IndexError:
            raise IndexError("One of the label is out of range. Got '{}'".format(label))
        return classes


def label_encoder(input_df, ordinal=None, nominal=None, exclude=None, inplace=False, drop_first=False,
                  custom_scale=None):
    """
    Convert categorical features in a dataframe to code value.
    All columns (i.e. feature) with dtype="object" will be considered as categorical and converted.
    Use 'exclude' to exclude "object" feature from the convertion
    Use 'nominal' or 'ordinal' to add non "object" type feature to the convertion.
    By default all categorical features (dtype="objecct") are considered nominal.
    Nominal features are converted using OneHotEncoding with 1st column dropped to avoid the Dummy Variable Trap
    Ordinal features are converted to a single column with code following the lexicographic order
    :param input_df:    [Pandas DataFrame] Input DataFrame
    :param nominal:     [List of str] Name of the nominal features (categorical feature with no ordering in their value)
    :param ordinal:     [List of str] Name of the ordinal features (categorical feature with order in their value)
    :param exclude:     [List of str] Name of the features to exclude from the conversion.
    :param inplace:     [Bool] If True modify the DataFrame in place.
    :param drop_first:  [Bool] If True, the 1st column of the one hot encoded dataframe is dropped to avoid
                        the dummy variables trap.
    :param custom_scale:[dict] Custom scale for odrinal feature. If None, order will be lexicographical. Ex:
                        {"feature_name": ["Excellent", "Average", "Bad"]}
    :return:            [Tuple: (pandas.DataFrame, dict)] Return the modified dataframe and the sklearn binarizer used
                        for the encoding in a dictionary of the following shape:
                        {"name_of_the_feature": encoder_function}
    """
    df = input_df if inplace else input_df.copy(deep=True)  # type: pd.DataFrame
    cat_feature = list(df.select_dtypes("object"))
    if ordinal is None:
        ordinal = []
    if nominal is None:
        nominal = []
    if custom_scale is None:
        custom_scale = []
    cat_feature = list(set(cat_feature + ordinal + nominal))
    if exclude is not None:
        for feature in exclude:
            try:
                cat_feature.pop(cat_feature.index(feature))
            except ValueError:
                pass
    labelizer = {}
    for feature in cat_feature:
        try:
            if df[feature].isnull().sum() > 0:
                raise ValueError(f"df contains NaN in feature '{feature}'. Please remove NaN before label encoding")
        except KeyError:
            continue
        if feature in ordinal:
            if feature in custom_scale:
                labelizer[feature] = LabelEncoder()
                labelizer[feature].fit(custom_scale[feature], order="keep")
            else:
                labelizer[feature] = preprocessing.LabelEncoder()
                labelizer[feature].fit(df[feature])
            df[feature] = labelizer[feature].transform(df[feature])
        else:
            labelizer[feature] = preprocessing.MultiLabelBinarizer()
            onehot = labelizer[feature].fit_transform(df[feature].values.reshape(-1, 1))
            col_name = [f"{feature}_{i}" for i in labelizer[feature].classes_]
            df_onehot = pd.DataFrame(onehot, columns=col_name, index=df.index)
            if drop_first is True:
                df_onehot = df_onehot.drop(f"{feature}_{labelizer[feature].classes_[0]}", axis=1)
            df[df_onehot.columns] = df_onehot
            df.drop(feature, axis=1, inplace=True)
    return df, labelizer


def encode_cat_feature(dataset, output_dataset, output_label_file=None, drop_first=False,
                       header=0, index_col=None, ordinal_feature=None, nominal_feature=None, exclude_feature=None):
    df = pd.read_csv(dataset, header=header, index_col=index_col)
    df, label = label_encoder(df, ordinal=ordinal_feature, nominal=nominal_feature, exclude=exclude_feature,
                              drop_first=drop_first)
    df.to_csv(output_dataset, index=False if index_col is None else True, header=False if header is None else True)
    if output_label_file is not None:
        with Path(output_label_file).open(mode='b') as fp:
            pickle.dump(label, fp)
