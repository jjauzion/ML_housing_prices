import sklearn.preprocessing as preprocessing
import pandas as pd
from pathlib import Path
import pickle


def label_encoder(input_df, ordinal=None, nominal=None, exclude=None, inplace=False):
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
    cat_feature += ordinal + nominal
    if exclude is not None:
        for feature in exclude:
            try:
                cat_feature.pop(cat_feature.index(feature))
            except ValueError:
                pass
    labelizer = {}
    for feature in cat_feature:
        if df[feature].isnull().sum() > 0:
            raise ValueError(f"dataframe contains NaN in feature '{feature}'. Please remove NaN before label encoding")
        if feature in ordinal:
            labelizer[feature] = preprocessing.LabelEncoder()
            df[feature] = labelizer[feature].fit_transform(df[feature])
        else:
            labelizer[feature] = preprocessing.MultiLabelBinarizer()
            onehot = labelizer[feature].fit_transform(df[feature].values.reshape(-1, 1))
            col_name = [f"{feature}_{i}" for i in labelizer[feature].classes_]
            df_onehot = pd.DataFrame(onehot, columns=col_name)
            df_onehot = df_onehot.drop(f"{feature}_{labelizer[feature].classes_[0]}", axis=1)
            df[df_onehot.columns] = df_onehot
            df.drop(feature, axis=1, inplace=True)
    return df, labelizer


def encode_cat_feature(dataset, output_dataset, output_label_file=None,
                       header=0, index_col=None, ordinal_feature=None, nominal_feature=None, exclude_feature=None):
    df = pd.read_csv(dataset, header=header, index_col=index_col)
    df, label = label_encoder(df, ordinal=ordinal_feature, nominal=nominal_feature, exclude=exclude_feature)
    df.to_csv(output_dataset, index=False if index_col is None else True)
    if output_label_file is not None:
        with Path(output_label_file).open(mode='b') as fp:
            pickle.dump(label, fp)