import pandas as pd
import argparse
from pathlib import Path

from src import toolbox

parse = argparse.ArgumentParser()
parse.add_argument("file", help="Configuration file, json format")
args = parse.parse_args()

data_conf = toolbox.DataConf()
try:
    data_conf.import_from_json(args.file)
except Exception as err:
    print(f"Error: can't read json configuration file '{Path(args.file)}' because: {err.__class__.__name__}:{err}")
    exit(0)

df = pd.read_csv(data_conf.raw_data_file)
null = df.isnull().sum()
percentage = null / df.isnull().count()
synthesis = pd.concat((null, percentage), axis=1, keys=("Total", "Percentage")).sort_values(ascending=False, by="Total")
synthesis = synthesis[synthesis["Total"] != 0]
print(synthesis)
