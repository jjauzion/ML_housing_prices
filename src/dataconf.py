import json
from pathlib import Path

from . import error_lib


class DataConf:

    def __init__(self):
        self.target_col = None
        self.raw_dataset = None
        self.cleaned_dataset = None
        self.nan_column_threshold = None
        self.cross_correlation_threshold = None
        self.feature_type = None
        self.header = None
        self.index_col = None
        self.sep = ","
        self.nan_values = ["#N/A", "#N/A", "N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
                           "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
        self.replace_nan = None
        self.ordinal_scale = None

    def __repr__(self):
        ret = ""
        for attribute in self.__dict__:
            ret += f'{attribute}: {getattr(self, attribute)}\n'
        return ret

    def import_from_json(self, file):
        try:
            with Path(file).open(mode='r', encoding='utf-8') as fp:
                conf = json.load(fp)
        except json.JSONDecodeError as err:
            raise error_lib.FileError(errtype=f"{err.__class__.__name__}", file=str(Path(file)), message=str(err))
        self.target_col = conf["target_col"]
        self.raw_dataset = conf["raw_dataset"]
        self.cleaned_dataset = conf["cleaned_dataset"]
        self.nan_column_threshold = conf["nan_column_threshold"]
        self.cross_correlation_threshold = conf["cross_correlation_threshold"]
        self.feature_type = conf["feature_type"]
        self.header = conf["header"]
        self.index_col = conf["index_col"]
        self.sep = conf["sep"]
        self.nan_values = conf["nan_values"]
        self.replace_nan = conf["replace_nan"]
        self.ordinal_scale = conf["ordinal_scale"]

    def to_json(self, file):
        with Path(file).open(mode='w') as fp:
            json.dump(self.__dict__, fp, indent=4, separators=(",", ": "))
        print(f'Dataconf saved to "{Path(file)}"')
