import sklearn.preprocessing as preprocessing
import pandas as pd


def label_encoder(input_df, nominal=None, ordinal=None, numeric=None, exclude=None, inplace=False):
    """
    Convert categorical features in a dataframe to code value.
    Nominal features are converted using OneHotEncoding with 1st column dropped to avoid the Dummy Variable Trap
    Ordinal features are converted to a single column with code following the lexicographic order
    By default all categorical features are considered nominal (i.e., non ordinal: label without value order)
    By default, numeric=None, only columns with dtype="object" will be converted.
    :param input_df:    [Pandas DataFrame] Input DataFrame
    :param nominal:     [List of str] Name of the nominal features.
    :param ordinal:     [List of str] Name of the ordinal features.
    :param numeric:     [List of str] Name of the features with numeric value that shall be considered as categorical.
    :param exclude:     [List of str] Name of the features to exclude from the conversion.
    :param inplace:     [Bool] If True modify the DataFrame in place.
    :return:
    """
    df = input_df if inplace else input_df.copy(deep=True)  # type: pd.DataFrame
    cat_feature = list(df.select_dtypes("object"))
    cat_feature += numeric
    for feature in exclude:
        try:
            cat_feature.pop(cat_feature.index(feature))
        except ValueError:
            pass
    cat_col_index = []
    labelizer = {}
    for feature in cat_feature:
        if feature in ordinal:
            labelizer[feature] = preprocessing.MultiLabelBinarizer()
            onehot = labelizer[feature].fit_transform(df[feature].values.reshape(-1, 1))
            df_onehot = pd.DataFrame(onehot, columns=[f"{feature}_{i}" for i in range(labelizer[feature].classes_)])
            df_onehot.drop(f"{feature}_{labelizer[feature].classes_[0]}")
            df = pd.concat((df, df_onehot), axis=1)
        else:
            labelizer[feature] = preprocessing.LabelEncoder()
            df[feature] = labelizer[feature].fit_transform(df[feature])
    # for feature in cat_feature:
        # cat_col_index.append(df.columns.get_loc(feature))
    # one_hot = preprocessing.OneHotEncoder(categorical_features=cat_col_index)
    # df =

