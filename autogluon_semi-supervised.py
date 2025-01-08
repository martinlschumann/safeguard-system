"""Pipeline utilizing Autogluon's built-in support for Semi-Supervised Learning."""
import argparse
import logging

import pandas
from autogluon.tabular import TabularPredictor
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch

from custom_datasets import get_custom_datasets, data_group_1, data_group_2, data_group_3
from linear_ensemble_basis import get_datasets_by_study, plus_or_minus_ag, evaluate


def linear_ensemble_semi_supervised(fetched_dataset: Bunch, openml_dataset: str, combined_df: DataFrame,
                                    ag_model_is_plus: str):
    """The main linear_ensemble code.

    :param fetched_dataset: the full dataset
    :param openml_dataset: the name of the openml dataset.
    :param combined_df: the data of the dataset as a DataFrame
    :param ag_model_is_plus: the location of the Autogluon model file
    """
    print(f"{openml_dataset=}")

    time_limit = 600  # seconds

    class_name = fetched_dataset.target_names[0]
    print(f"{class_name=}")

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
    print(length_of_data)
    print(f"{split_labeled=}, {split_unlabeled=}")

    # split X and y into model 1 and model 2 and the test data to gauge performance:
    X_train_1 = X_train.iloc[:split_labeled, :]
    X_train_2 = X_train.iloc[split_labeled:split_unlabeled, :].drop(columns=[class_name])
    print(f"{X_train_1.shape} {X_train_2.shape}, {X_test.shape}")

    """# Run First Predictor"""
    hyperparameters = {
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
        "FT_TRANSFORMER": {}
    }

    try:
        predictor1 = TabularPredictor(label=class_name).fit(X_train_1, time_limit=time_limit, unlabeled_data=X_train_2,
                                                            hyperparameters=hyperparameters, presets='high_quality')
    # FIXME: error AttributeError: 'NoneType' object has no attribute 'dls'
    except AttributeError as e:
        print(e)
        return

    """# Linear Ensemble"""
    if not predictor1.can_predict_proba:
        print("can't predict probabilities")
        return
    y_pred_probabilities = predictor1.predict_proba(X_train_2)

    print(y_pred_probabilities.shape)
    # other strategies other than any column > min_confidence (statistical analysis)
    y_pred_2 = y_pred_probabilities[y_pred_probabilities.gt(min_confidence).any(axis=1)].apply('idxmax', axis=1)
    print(y_pred_2.shape)
    amount_ignored = y_pred_probabilities.shape[0] - y_pred_2.shape[0]
    print(f"{amount_ignored} ignored")
    values_ignored = y_pred_probabilities[y_pred_probabilities.lt(min_confidence).all(axis=1)]
    print(values_ignored.shape)
    logging.debug(values_ignored[:10])

    # so that original stays the same
    X_train_2_new = X_train_2.copy()
    X_train_2_new = X_train_2_new.join(y_pred_2.rename(class_name), how='inner')
    X_train_2_new = pandas.concat([X_train_1, X_train_2_new])

    print(f"{X_train_1.shape} {X_train_2_new.shape}, {X_test.shape}")
    print(f"{X_train_2.shape=}")

    """# Run Second Predictor"""

    is_plus = plus_or_minus_ag(fetched_dataset, X_train_1, ag_model_is_plus, groups="clustering")

    predictor2 = TabularPredictor(label=class_name).fit(X_train_2_new, time_limit=time_limit, presets='high_quality')

    """# Evaluate Predictors"""

    eval_predictor_1 = evaluate(predictor1, X_test)
    accuracy_1 = eval_predictor_1['accuracy']

    eval_predictor_2 = evaluate(predictor2, X_test)
    accuracy_2 = eval_predictor_2['accuracy']

    print(f"{eval_predictor_1=}")
    print(f"paper_version: {1 - accuracy_1}")
    print(f"{eval_predictor_2=}")
    print(f"paper_version: {1 - accuracy_2}")
    if accuracy_1 < accuracy_2:
        print("Accuracy is better")

    print(f"{is_plus=}")

    """# Show Leaderboards"""

    # predictor1.leaderboard(X_test, silent=True)
    # predictor2.leaderboard(X_test, silent=True)

    print("\n\n")


def main_100(ag_model_is_plus):
    """main to run co_ensemble using the OpenML100 dataset

    :param ag_model_is_plus: the autogluon model to calculate is_plus
    """
    for fetched_dataset, openml_dataset, combined_df in get_datasets_by_study(study_id=14):
        try:
            linear_ensemble_semi_supervised(fetched_dataset, openml_dataset, combined_df, ag_model_is_plus)
        except Exception as e:
            print(e)


def main_cc18(ag_model_is_plus):
    """main to run co_ensemble using the OpenML CC18 dataset

    :param ag_model_is_plus: the autogluon model to calculate is_plus
    """
    for fetched_dataset, openml_dataset, combined_df in get_datasets_by_study(study_id=99):
        try:
            linear_ensemble_semi_supervised(fetched_dataset, openml_dataset, combined_df, ag_model_is_plus)
        except Exception as e:
            print(e)


def main_cd(ag_model_is_plus, data_group):
    """main to run co_ensemble using custom datasets

    :param ag_model_is_plus: the autogluon model to calculate is_plus
    :param data_group: which custom dataset group to run.
    """
    for fetched_dataset, openml_dataset, combined_df in get_custom_datasets(data_group):
        try:
            linear_ensemble_semi_supervised(fetched_dataset, openml_dataset, combined_df, ag_model_is_plus)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='autogluon model folder name for is_plus', required=True)
    parser.add_argument('--dataset', '-d', type=str, choices=['100', 'CC18', 'CD1', 'CD2', 'CD3'],
                        help="which dataset to run", required=True)
    args = parser.parse_args()
    ag_model_is_plus = args.file
    dataset = args.dataset

    if dataset == '100':
        main_100(ag_model_is_plus)
    elif dataset == 'CC18':
        main_cc18(ag_model_is_plus)
    elif dataset == "CD1":
        main_cd(ag_model_is_plus, data_group_1)
    elif dataset == "CD2":
        main_cd(ag_model_is_plus, data_group_2)
    elif dataset == "CD3":
        main_cd(ag_model_is_plus, data_group_3)
    else:
        raise ValueError("Incorrect dataset value given")
