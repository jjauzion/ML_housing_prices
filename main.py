import pandas as pd

from src import preprocess_feature
from src import processing
from src import utils


def print_(string, verbosity):
    if verbosity > 0:
        print(string)


if __name__ == "__main__":
    args = utils.parse_main_args("file", "conf_output", "force", "verbosity", "edit")
    output_conf = args.conf_output if args.conf_output is not None else args.file
    try:
        dataset, df_clean, label, scaleX, scaleY = utils.import_df_from_dataconf(args.file, drop=True, transform=True,
                                                                                 encode=True, scale_data=True,
                                                                                 verbosity=args.verbosity)
        X_train, y_train, X_test, y_true = preprocess_feature.split_dataset(df_clean, dataset.target_col, ratio=0.8,
                                                                            seed=None)
        price_true = scaleY.inverse_transform(y_true)
        # Linear Reg
        _, y_pred = processing.fit_predict(X_train, y_train, X_test, model="LinearReg")
        print_(f'Result:\n{pd.concat((y_pred, y_true), axis=1).sort_values(by="LinearReg")}', args.verbosity)
        price_pred_1 = scaleY.inverse_transform(y_pred)
        print_(f'Result:\n{pd.concat((price_pred_1, price_true), axis=1)}', args.verbosity)
        test_score_1 = processing.rmsle_score(price_true, price_pred_1, zero_truncature=True)
        print_(f'Test score = {test_score_1}', args.verbosity)
        # Ridge
        _, y_pred = processing.fit_predict(X_train, y_train, X_test, model="Ridge", seed=0)
        print_(f'Result:\n{pd.concat((y_pred, y_true), axis=1).sort_values(by="Ridge")}', args.verbosity)
        price_pred = scaleY.inverse_transform(y_pred)
        print_(f'Result:\n{pd.concat((price_pred, price_true), axis=1)}', args.verbosity)
        test_score = processing.rmsle_score(price_true, price_pred, zero_truncature=True)
        print_(f'Test score = {test_score}', args.verbosity)
        print_(f'diff score = {test_score_1 - test_score}', verbosity=args.verbosity)
    except IOError as err:
        print(f'Error: {err}')
        exit(0)

