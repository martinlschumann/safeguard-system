#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Trains an Autogluon based Safeguard System Predictor"""

import argparse
import pprint

from autogluon.tabular import TabularPredictor, TabularDataset
from sklearn.model_selection import train_test_split

from autogluon_custom_metrics import ag_weighted_false_negative_loss_custom_scorer, ag_f_beta_custom_scorer
from linear_ensemble_basis import evaluate


def main(csv_file_name: str, eval_metric="recall"):
    """The main function for calculating

    :param csv_file_name: the csv file containing the meta-features to train on
    :param eval_metric: which evaluation metric to use to calculate improvements in training
    """
    combined_df = TabularDataset(csv_file_name)

    time_limit = 600  # seconds

    class_name = 'class'
    print(f"{class_name=}")

    # how much of the data is used for training
    training_ratio = 0.9
    # how much of training data stays labeled
    labeled_ratio = 1

    X_train, X_test = train_test_split(combined_df, test_size=1 - training_ratio, random_state=22)

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

    # set positive class to plus, so that recall works correctly.
    # recall tries to minimize the amount of false negatives, so it is a good metric,
    # as it minimizes the amount of fits that are not done but should be, as it would improve performance.
    predictor1 = TabularPredictor(label=class_name, eval_metric=eval_metric,
                                  learner_kwargs={"positive_class": "plus"}).fit(X_train_1, time_limit=time_limit,
                                                                                 presets='best_quality')

    y_pred_probabilities = predictor1.predict_proba(X_test)
    print("\n\n")
    pprint.pp(y_pred_probabilities)
    print("\n\n")

    """# Evaluate Predictors"""

    eval_predictor_1 = evaluate(predictor1, X_test)
    accuracy_1 = eval_predictor_1['accuracy']
    print(f"{eval_predictor_1=}")
    print(f"paper_version: {1 - accuracy_1}")
    print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='csv file name', required=True)
    args = parser.parse_args()

    # main(args.file)
    main(args.file, ag_f_beta_custom_scorer)
    # main(args.file, ag_weighted_false_negative_loss_custom_scorer)
