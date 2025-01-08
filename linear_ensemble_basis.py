"""Autogluon Co Ensembling Code
For Predicting plus or minus, can use AG's TabularPredictor or sklearns' RandomForestClassifier
Also includes some helper functions.
"""
import logging

from read_out_txt import OpenMLDatasetResult

try:
    import openml
    from autogluon.tabular import TabularPredictor
except ImportError as e:
    openml = None
    TabularPredictor = None

import pandas
from joblib import load
from pandas import DataFrame
from pymfe.mfe import MFE
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import Bunch

import signal
from contextlib import contextmanager


# from https://stackoverflow.com/a/601168
@contextmanager
def time_limit(seconds: int):
    """Runs a function for a given number of seconds

    :param seconds: the time limit
    """

    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def evaluate(predictor, X_test):
    """Calls `predictor.evaluate` with silent=True

    :param predictor: the predictor
    :param X_test: the data to test on
    :return: the result of predictor.evaluate
    """
    return predictor.evaluate(X_test, silent=True)


class OutputAppender():
    """Helper class for saving the output from linear ensembling"""
    output = []

    def append(self, output_to_append):
        """Append to the output list.
        Adds a newline, so it works like a print statement.

        :param output_to_append: the output to append
        """
        self.output.append(str(output_to_append) + "\n")
        # NOTE: print when adding?

    def __str__(self) -> str:
        return "".join(self.output)

    def openml_dataset(self) -> OpenMLDatasetResult:
        """Create an OpenMLDatasetResult from the output

        :return: OpenMLDatasetResult
        """
        return OpenMLDatasetResult.parse(self.output)

    def json(self) -> str:
        """Create an OpenMLDatasetResult json string

        :return: OpenMLDatasetResult as json
        """
        return OpenMLDatasetResult.parse(self.output).to_json()

    def print(self):
        """Print the full output, with no added newline."""
        print(str(self), end="")


def plus_or_minus_ag(fetched_dataset: Bunch, combined_df: DataFrame, ag_model_file: str,
                     groups: list[str] | str = None) -> bool | None:
    """Calculate whether safeguard system predicts plus or minus with an autogluon based classifier.

    :param fetched_dataset: the full dataset
    :param combined_df: the data of the dataset as a DataFrame
    :param ag_model_file: the location of the Autogluon model file
    :param groups: which pymfe groups to utilize
    :return: True or False if no errors, otherwise None
    """
    predictor = TabularPredictor.load(ag_model_file)
    return _plus_or_minus(fetched_dataset, combined_df, predictor=predictor, groups=groups)


def _plus_or_minus(fetched_dataset: Bunch, combined_df: DataFrame,
                   predictor: RandomForestClassifier | TabularPredictor, groups: list[str] | str = None) -> bool | None:
    """Internal function for the safeguard system which predicts plus or minus.

    :param fetched_dataset: the full dataset
    :param combined_df: the data of the dataset as a DataFrame
    :param predictor: the predictor object, which is either a RandomForestClassifier or a TabularPredictor
    :param groups: which pymfe groups to utilize (if None, it is set to model-based)
    :return: True or False if no errors, otherwise None
    """
    if groups is None:
        groups = ["model-based"]
        print("groups was None, set to model-based")

    class_name = fetched_dataset.target_names[0]

    # mfe requires list or numpy array
    X = combined_df.drop(columns=[class_name]).to_numpy()
    y = combined_df[class_name].to_numpy()

    mfe = MFE(groups=groups)
    try:
        # sometimes fit takes too long to run, so limit it to 10 minutes of runtime
        with time_limit(10 * 60):  # 10 minutes, same as Autogluon run
            mfe.fit(X, y)
            ft = mfe.extract(out_type=DataFrame)
    except (ValueError, RecursionError, IndexError, TimeoutError) as e:
        # IndexError e.g. for sylva_agnostic
        # missing values does not work with MFE
        print(e)
        return None

    try:
        prediction = predictor.predict(ft)
    except (KeyError, ValueError) as e:
        # some datasets don't contain the relevant metafeatures, e.g. sylva_agnostic
        # some datasets can be to large or contain infinite values (e.g. SEA(50))
        print(e)
        return None

    if prediction.item() == "plus":
        return True
    return False


def plus_or_minus_sklearn(fetched_dataset: Bunch, combined_df: DataFrame,
                          sklearn_model_is_plus_file: str, groups: list[str] | str = None) -> bool | None:
    """Calculate whether safeguard system predicts plus or minus with an RandomForestClassifier based classifier.

    :param fetched_dataset: the full dataset
    :param combined_df: the data of the dataset as a DataFrame
    :param sklearn_model_is_plus_file: the location of the RandomForestClassifier model file
    :param groups: which pymfe groups to utilize
    :return: True or False if no errors, otherwise None
    """
    clf: RandomForestClassifier = load(sklearn_model_is_plus_file)
    return _plus_or_minus(fetched_dataset, combined_df, predictor=clf, groups=groups)


def get_datasets_by_study(start_index=0, study_id=14) -> list[tuple[Bunch, str, DataFrame]]:
    """Get dataset by study id from openml.org

    openml 100 from: https://docs.openml.org/benchmark/#openml100
    has the id 14, instead of 99
    99 is the newer benchmark version: OpenML-CC18 Curated Classification benchmark

    :param start_index: from which dataset to start
    :param study_id: the study id, according to openml.org
    :return: the datasets as a tuple of data, openml_name, and data as a dataframe,
        shuffled with random state=42
    """

    suite = openml.study.get_suite(study_id)
    datasets: list[tuple[Bunch, str, DataFrame]] = []
    index_counter = 0
    for task in suite.tasks:
        try:
            if index_counter >= start_index:
                # to fix studies not being found
                # task id and data_id are not always the same
                data_id = openml.tasks.get_task(task).dataset_id
                # set parser to silence warning
                # NOTE: a few studies are sparse data, can't use as_frame, just get ignored for now
                fetched_dataset = fetch_openml(data_id=data_id, as_frame=True, parser='liac-arff')
                name = fetched_dataset.details["name"]
                combined_df = fetched_dataset.frame.sample(frac=1, random_state=42)
                datasets.append((fetched_dataset, name, combined_df))
        except Exception as e:
            print(f"Could not get task with id: {task}, error: {e}")
        index_counter += 1
        print(f"{index_counter=}")
    print(f"{len(datasets)=}")
    return datasets


def linear_ensemble(fetched_dataset: Bunch, openml_dataset: str, combined_df: DataFrame, ag_model_is_plus: str = None,
                    sklearn_model_is_plus: str = None, groups: list[str] | str = None, verbose: bool = True,
                    safeguard: bool = False) -> (OutputAppender, TabularPredictor):
    """The main linear ensemble process.

    :param fetched_dataset: the full dataset
    :param openml_dataset: the name of the openml dataset.
    :param combined_df: the data of the dataset as a DataFrame
    :param ag_model_is_plus: the location of the Autogluon model file (only one can be non-None)
    :param sklearn_model_is_plus: the location of the RandomForestClassifier model file (only one can be non-None)
    :param groups: which pymfe groups to utilize (default is model-based if none given)
    :param verbose: whether the output should be printed at the end of the linear_ensemble run.
    :param safeguard: whether the safeguard system should stop after training M1 if M2 is predicted to perform worse.
    :return: the output as an OutputAppender, and either the M1 or M2 model.
    """
    oa: OutputAppender = OutputAppender()
    oa.append(f"{openml_dataset=}")

    time_limit = 600  # seconds

    class_name = fetched_dataset.target_names[0]
    oa.append(f"{class_name=}")

    # how much of the data is used for training
    training_ratio = 0.8
    # how much of training data stays labeled
    labeled_ratio = 0.1

    # the minimum threshold a label has to have for the data item to be selected
    min_confidence = 0.8

    X_train, X_test = train_test_split(combined_df, test_size=1 - training_ratio, random_state=42)

    length_of_data = X_train.shape[0]
    split_labeled = int(length_of_data * labeled_ratio)
    split_unlabeled = split_labeled + int(length_of_data * (1 - labeled_ratio))
    oa.append(length_of_data)
    oa.append(f"{split_labeled=}, {split_unlabeled=}")

    # split X and y into model 1 and model 2 and the test data to gauge performance:
    X_train_1 = X_train.iloc[:split_labeled, :]
    X_train_2 = X_train.iloc[split_labeled:split_unlabeled, :].drop(columns=[class_name])
    oa.append(f"{X_train_1.shape} {X_train_2.shape}, {X_test.shape}")

    """# Run First Predictor"""
    predictor1 = TabularPredictor(label=class_name).fit(X_train_1, time_limit=time_limit, presets="high_quality")

    """# Run Safeguard System"""

    # NOTE: instead of combined_df, use X_train_1?
    if ag_model_is_plus is not None:
        is_plus = plus_or_minus_ag(fetched_dataset, X_train_1, ag_model_is_plus, groups)
    elif sklearn_model_is_plus is not None:
        is_plus = plus_or_minus_sklearn(fetched_dataset, X_train_1, sklearn_model_is_plus, groups)
    else:
        # NOTE: leave this here?
        oa.print()
        raise Exception("both models can't be None")

    if safeguard:
        # if is_plus is None or True, continue
        if is_plus is False:
            oa.append(f"{is_plus=}")
            oa.append("\n\n")
            oa.print()
            return oa, predictor1

    """# Linear Ensemble"""
    if not predictor1.can_predict_proba:
        oa.print()
        print("can't predict probabilities")
        print("\n\n")
        return oa, predictor1

    y_pred_probabilities = predictor1.predict_proba(X_train_2)

    oa.append(y_pred_probabilities.shape)
    # other strategies other than any column > min_confidence (statistical analysis)
    y_pred_2 = y_pred_probabilities[y_pred_probabilities.gt(min_confidence).any(axis=1)].apply('idxmax', axis=1)
    oa.append(y_pred_2.shape)
    amount_ignored = y_pred_probabilities.shape[0] - y_pred_2.shape[0]
    oa.append(f"{amount_ignored} ignored")
    values_ignored = y_pred_probabilities[y_pred_probabilities.lt(min_confidence).all(axis=1)]
    oa.append(values_ignored.shape)
    logging.debug(values_ignored[:10])

    # so that original stays the same
    X_train_2_new = X_train_2.copy()
    X_train_2_new = X_train_2_new.join(y_pred_2.rename(class_name), how='inner')
    X_train_2_new = pandas.concat([X_train_1, X_train_2_new])

    oa.append(f"{X_train_1.shape} {X_train_2_new.shape}, {X_test.shape}")
    oa.append(f"{X_train_2.shape=}")

    """# Run Second Predictor"""

    predictor2 = TabularPredictor(label=class_name).fit(X_train_2_new, time_limit=time_limit, presets="high_quality")

    """# Evaluate Predictors"""

    eval_predictor_1 = evaluate(predictor1, X_test)
    accuracy_1 = eval_predictor_1['accuracy']

    eval_predictor_2 = evaluate(predictor2, X_test)
    accuracy_2 = eval_predictor_2['accuracy']

    oa.append(f"{eval_predictor_1=}")
    oa.append(f"paper_version: {1 - accuracy_1}")
    oa.append(f"{eval_predictor_2=}")
    oa.append(f"paper_version: {1 - accuracy_2}")
    if accuracy_1 < accuracy_2:
        oa.append("Accuracy is better")

    oa.append(f"{is_plus=}")

    oa.append("\n\n")

    if verbose:
        oa.print()

    if is_plus is False:
        return oa, predictor1
    return oa, predictor2
