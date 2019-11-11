import argparse


def is_positive(value):
    f_value = float(value)
    if f_value < 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive value.".format(value))
    return f_value


def is_percentage(value):
    f_value = float(value)
    if f_value < 0 or f_value > 1:
        raise argparse.ArgumentTypeError("'{}' is not a valid percentage value (should be between 1 and 0).".format(value))
    return f_value


def is_positive_int(value):
    i_value = int(value)
    if i_value < 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive int value.".format(value))
    return i_value


def is_strick_positive_int(value):
    i_value = int(value)
    if i_value <= 0:
        raise argparse.ArgumentTypeError("'{}' is not a valid positive int value.".format(value))
    return i_value
