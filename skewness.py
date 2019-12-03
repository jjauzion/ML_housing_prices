import pandas as pd
import numpy as np
from scipy import stats

from src import features_selection
from src import dataconf
from src import utils


def print_(string, verbosity):
    if verbosity > 0:
        print(string)


if __name__ == "__main__":
    args = utils.parse_main_args("file", "conf_output", "verbosity", "transform", "encode", "scale", "drop", "unskew")
    output_conf = args.conf_output if args.conf_output is not None else args.file
    dataset, df, _, _, _ = utils.import_df_from_dataconf(args.file,
                                                         drop=args.drop,
                                                         encode=args.encode,
                                                         transform=args.transform,
                                                         scale=args.standardize,
                                                         unskew=args.unskew)
    print_("Skewness analysis".center(40, "-"), args.verbosity)
    numeric_feat = df.dtypes[df.dtypes != "object"].index
    skewed_feat = df[numeric_feat].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
    print("\nSkew in numerical features: \n")
    skewness = pd.DataFrame({'Abs Skewness': skewed_feat}).abs()
    end = False
    threshold = dataset.skewness_threshold
    skewness["Box Cox Transform"] = skewness["Abs Skewness"] > threshold
    while not end:
        print(skewness)
        print("\nDo you want to apply Box Cox transformation to the feature selected above?")
        end = utils.prompt_validation_or_new_threshold(min_=0, max_=np.inf)
        if end == "exit":
            exit(0)
        if end != "yes" and end is not None:
            threshold = float(end)
            skewness["Box Cox Transform"] = skewness["Abs Skewness"] > threshold
            end = False
    unskew_feature = skewness[skewness["Box Cox Transform"]].index.to_list()
    dataset.unskew += [feature for feature in unskew_feature if feature not in dataset.unskew]
    dataset.to_json(output_conf, verbosity=args.verbosity)
    if args.verbosity > 0:
        print('\n--------------------------\nFeature to "unskew" updated!')
        print(f'New list of feature to unskew: {dataset.unskew}')
        print(f'Data configuration file saved to "{output_conf}"')
