import json
from pathlib import Path

from . import error_lib


class DataConf:

    def __init__(self):
        # Required attribute (NOTE: not up to date)
        self.target_col = None
        self.raw_dataset = None
        self.cleaned_dataset = None
        self.nan_column_threshold = None
        self.cross_correlation_threshold = None
        self.feature_type = None
        self.header = None
        self.index_col = None
        self.replace_nan = None
        self.ordinal_scale = None
        self.useless_line = None
        self.transform = None
        # Optional attribute
        self.skewness_threshold = 0.9
        self.unskew = []
        self.sep = ","
        self.nan_values = ["#N/A", "#N/A", "N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
                           "N/A", "NA", "NULL", "NaN", "n/a", "nan", "null"]
        self.useless = "useless"

    def __repr__(self):
        ret = ""
        for attribute in self.__dict__:
            ret += f'{attribute}: {getattr(self, attribute)}\n'
        return ret

    def import_from_json(self, file):
        def get_val(attribute, conf, key, key_error="raise"):
            try:
                return conf[key]
            except KeyError as err:
                if key_error == "raise":
                    raise err
                else:
                    return getattr(self, attribute)
        try:
            with Path(file).open(mode='r', encoding='utf-8') as fp:
                conf = json.load(fp)
        except json.JSONDecodeError as err:
            raise error_lib.FileError(errtype=f"{err.__class__.__name__}", file=str(Path(file)), message=str(err))
        self.target_col = get_val("target_col", conf, "target_col")
        self.raw_dataset = get_val("raw_dataset", conf, "raw_dataset")
        self.cleaned_dataset = get_val("cleaned_dataset", conf, "cleaned_dataset")
        self.nan_column_threshold = get_val("nan_column_threshold", conf, "nan_column_threshold")
        self.cross_correlation_threshold = get_val("cross_correlation_threshold", conf, "cross_correlation_threshold")
        self.feature_type = get_val("feature_type", conf, "feature_type")
        self.header = get_val("header", conf, "header")
        self.index_col = get_val("index_col", conf, "index_col")
        self.replace_nan = get_val("replace_nan", conf, "replace_nan")
        self.ordinal_scale = get_val("ordinal_scale", conf, "ordinal_scale")
        self.useless_line = get_val("useless_line", conf, "useless_line")
        self.transform = get_val("transform", conf, "transform")
        self.skewness_threshold = get_val("skewness_threshold", conf, "skewness_threshold", key_error="ignore")
        self.unskew = get_val("unskew", conf, "unskew", key_error="ignore")
        self.sep = get_val("sep", conf, "sep", key_error="ignore")
        self.nan_values = get_val("nan_values", conf, "nan_values", key_error="ignore")

    def to_json(self, file, verbosity=1):
        with Path(file).open(mode='w') as fp:
            json.dump(self.__dict__, fp, indent=4, separators=(",", ": "))
        if verbosity > 0:
            print(f'Dataconf saved to "{Path(file)}"')

    def change_feature_type(self, feature_list, new_type):
        """
        Change the type of all feature in feature_list to new_type.
        If the feature is already in one (or more) self.feature_type list, it will be removed from the list and appended
        to the new_type list.
        NOTE if there is the feature appears several time in the same feature_type, only the first value will be removed
        :param feature_list:    [List] list of feature to change to new_type
        :param new_type:        [str] name of the new_type (shall already exist in self.feature_type)
                                If new_type is None, the feature will just be removed from previous type (if any)
        """
        for feature in feature_list:
            for type_ in self.feature_type:
                try:
                    self.feature_type[type_].remove(feature)
                except ValueError:
                    pass
            if new_type is not None:
                self.feature_type[new_type].append(feature)

    def delete_duplicate(self):
        for type_ in self.feature_type:
            self.feature_type[type_] = list(set(self.feature_type[type_]))
        self.useless_line = list(set(self.useless_line))

    def add_useless_line(self, line):
        try:
            iterator = line.__iter__()
            for l in iterator:
                if l not in self.useless_line:
                    self.useless_line.append(l)
        except AttributeError:
            if line not in self.useless_line:
                self.useless_line.append(line)
