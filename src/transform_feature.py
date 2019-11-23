import pandas as pd


def transform_feature(dataframe, transform_dic, delete_original=False):
    """
    Create new features from existing feature in the dataframe based on the transformation_dict dictionary.
    :param dataframe:
    :param transform_dic:
    :param delete_original:
    :return:
    """
    for feature, trans in transform_dic.items():
        if trans["operator"] == "+":
            dataframe[feature] = dataframe[trans["from"]].sum(axis=1)
        elif trans["operator"] == "-":
            dataframe[feature] = dataframe[trans["from"][0]]
            for feat in trans["from"][1:]:
                dataframe[feature] = dataframe[feature].sub(dataframe[feat])


if __name__ == "__main__":
    df = pd.read_csv("../data/train_cleaned.csv", index_col=0)
    l = ["7", "5"]
    dic = {
        "f1": {
            "from": l,
            "operator": "+"
        },
        "f2": {
            "from": l,
            "operator": "-"
        }
    }
    transform_feature(df, dic)
    print(df)

