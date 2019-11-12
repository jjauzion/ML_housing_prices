import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

from src import toolbox

parse = argparse.ArgumentParser()
parse.add_argument("file", help="Configuration file, json format")
parse.add_argument("-o", "--output", default="data/dataset_cleaned.csv", type=str,
                   help="Output file for the cleaned dataset")
parse.add_argument("-yf", "--force", action="store_true", help="Force 'yes' answer to any prompt")
args = parse.parse_args()

data_conf = toolbox.DataConf()
try:
    data_conf.import_from_json(args.file)
except Exception as err:
    print(f"Error: can't read json configuration file '{Path(args.file)}' because: {err.__class__.__name__}:{err}")
    exit(0)
data_conf = Path("data/train.csv")
threshold = 0.9
target_col = 81
non_informative_col = []
output_file = Path("data/data_cleaned.csv")


target_col_l = str(target_col) + "_l"
df = pd.read_csv(data_file, header=None, names=list(range(32)))
df[target_col] = pd.Categorical(df[target_col])
df[target_col_l] = df[target_col].cat.codes
corr = df.drop(columns=target_col).corr()
feature_corr = corr.drop(target_col_l)
feature_corr = feature_corr.drop(target_col_l, axis=1)
target_corr = corr[target_col_l].drop(target_col_l)
corr_list = []
for i, col in feature_corr.iteritems():
    for j, val in col.iteritems():
        if j >= i:
            feature_corr[i][j] = 0
        else:
            corr_list.append(feature_corr[i][j])
top_cor = feature_corr.unstack().sort_values(ascending=False)
top_cor = top_cor[top_cor > threshold]
col2del = non_informative_col
for elm in top_cor.index:
    if elm[0] not in col2del and elm[1] not in col2del:
        if target_corr[elm[0]] < target_corr[elm[1]]:
            col2del.append(elm[0])
        else:
            col2del.append(elm[1])
if args.verbose > 0:
    print("Feature correlation to the target:\n", target_corr)
    print("Correlated features:\n", top_cor)
    print("List of column that should be deleted: {}".format(col2del))
if args.verbose > 1:
    fig1 = plt.figure()
    target_corr.sort_values().plot.bar()
    plt.title("Features correlation to the target")
    plt.xlabel("Features")
    plt.ylabel("Correlation coeff value")
    tmp = pd.DataFrame(corr_list)
    tmp.plot.hist()
    plt.title("Features cross correlation coefficient")
    plt.xlabel("Correlation coeff value")
    fig2 = plt.figure()
    sns.heatmap(feature_corr, square=True, linewidths=0.5, linecolor="Black", fmt=".1f", annot=True, cbar_kws={"shrink":0.70}, vmax=1, center=0, vmin=-1, cmap="PiYG")
    plt.title("Cross correlation matrix")
    plt.show()
df_out = df.drop(columns=col2del)
df_out = df_out.drop(columns=target_col_l)
df_out.to_csv(output_file, sep=',', index=False)
if args.verbose > 0:
    print("cleaned data saved to '{}'".format(output_file))
