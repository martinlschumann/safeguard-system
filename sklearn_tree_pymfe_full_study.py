#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run through the full dataset(s) selected with the full linear-ensembling pipeline."""
import argparse
from typing import Callable

from pandas import DataFrame
from sklearn.utils import Bunch

from custom_datasets import get_custom_datasets, data_group_1, data_group_2, data_group_3
from linear_ensemble_basis import get_datasets_by_study, linear_ensemble


def main(sklearn_model_is_plus, dataset_getter: Callable[[], list[tuple[Bunch, str, DataFrame]]]):
    for fetched_dataset, openml_dataset, combined_df in dataset_getter():
        try:
            linear_ensemble(fetched_dataset, openml_dataset, combined_df, sklearn_model_is_plus=sklearn_model_is_plus,
                            groups=["model-based", "clustering"])
        except Exception as e:
            print(f"problem with {openml_dataset}, error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='sklearn model joblib name', required=True)
    parser.add_argument('-d', '--dataset', choices=["100", "CC", "CD1", "CD2", "CD3"],
                        help='dataset to use, possible values: 100, CC, CD1, CD2, CD3',
                        required=True)
    args = parser.parse_args()
    sklearn_model_is_plus = args.file

    dataset_mapping: dict[str, Callable[[], list[tuple[Bunch, str, DataFrame]]]] = {
        "100": lambda: get_datasets_by_study(0, study_id=14),
        "CC": lambda: get_datasets_by_study(0, study_id=99),
        "CD1": lambda: get_custom_datasets(data_group_1),
        "CD2": lambda: get_custom_datasets(data_group_2),
        "CD3": lambda: get_custom_datasets(data_group_3)}

    dataset_getter: Callable[[], list[tuple[Bunch, str, DataFrame]]] = dataset_mapping[args.dataset]

    main(sklearn_model_is_plus, dataset_getter)
