#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run linear ensembling on Custom Datasets 1-3"""
import argparse

from linear_ensemble_basis import linear_ensemble
from custom_datasets import get_custom_datasets, data_group_1, data_group_2, data_group_3


def main(ag_model_is_plus, data_group):
    for fetched_dataset, openml_dataset, combined_df in get_custom_datasets(data_group):
        try:
            linear_ensemble(fetched_dataset, openml_dataset, combined_df, ag_model_is_plus, groups=["model-based"])
        # FIXME: catch all exceptions for now
        except Exception as e:
            # NOTE: for regression problems, find different strategy
            print(f"problem with {openml_dataset}, error: {e}")
            print("\n\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='autogluon model folder name', required=True)
    parser.add_argument('-d', '--data-group', type=int, help='which data group to run', required=True)
    args = parser.parse_args()
    ag_model_is_plus = args.file

    if args.data_group == 1:
        main(ag_model_is_plus, data_group_1)
    elif args.data_group == 2:
        main(ag_model_is_plus, data_group_2)
    elif args.data_group == 3:
        main(ag_model_is_plus, data_group_3)
    else:
        print(f"{args.data_group} can only be 1, 2 or 3")
