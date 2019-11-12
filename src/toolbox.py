import json
from pathlib import Path

from . import error_lib


class DataConf:

    def __init__(self):
        self.tcol = None
        self.raw_data_file = None
        self.nan_filter_file = None
        self.feature_selec_file = None
        self.cross_correlation_threshold = None

    def import_from_json(self, file):
        try:
            with Path(file).open(mode='r', encoding='utf-8') as fp:
                conf = json.load(fp)
        except json.JSONDecodeError as err:
            raise error_lib.FileError(errtype=f"{err.__class__.__name__}", file=str(Path(file)), message=str(err))
        self.tcol = conf["target_col"]
        self.raw_data_file = Path(conf["raw_data_file"])
        self.nan_filter_file = Path(conf["nan_filter_file"])
        self.feature_selec_file = Path(conf["feature_selec_file"])
        self.cross_correlation_threshold = conf["cross_correlation_threshold"]

    def __repr__(self):
        ret = f"Target column = {self.tcol}\n"
        ret += f"raw_data_file = {self.raw_data_file}\n"
        ret += f"nan_filter_file = {self.nan_filter_file}\n"
        ret += f"feature_selec_file = {self.feature_selec_file}\n"
        ret += f"cross_correlation_threshold = {self.cross_correlation_threshold}\n"
        return ret

