import json
from pathlib import Path

from . import error_lib


class DataConf:

    def __init__(self):
        self.tcol = None
        self.raw_dataset = None
        self.cleaned_dataset = None
        self.nan_column_threshold = None
        self.cross_correlation_threshold = None
        self.feature_type = None
        self.header = None
        self.index_col = None

    def import_from_json(self, file):
        try:
            with Path(file).open(mode='r', encoding='utf-8') as fp:
                conf = json.load(fp)
        except json.JSONDecodeError as err:
            raise error_lib.FileError(errtype=f"{err.__class__.__name__}", file=str(Path(file)), message=str(err))
        self.tcol = conf["target_col"]
        self.raw_dataset = Path(conf["raw_dataset"])
        self.cleaned_dataset = Path(conf["cleaned_dataset"])
        self.nan_column_threshold = conf["nan_column_threshold"]
        self.cross_correlation_threshold = conf["cross_correlation_threshold"]
        self.feature_type = conf["feature_type"]
        self.header = conf["header"]
        self.index_col = conf["index_col"]

    def __repr__(self):
        ret = ""
        for attribute in self.__dict__:
            ret += f'{attribute}: {getattr(self, attribute)}\n'
        return ret

