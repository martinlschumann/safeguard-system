#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run linear ensembling on OpenMLCC18"""
import argparse

from linear_ensemble_basis import get_datasets_by_study, linear_ensemble


def main(ag_model_is_plus):
    start_index = 0

    # openml 100 from: https://docs.openml.org/benchmark/#openml100
    # has the id 14, instead of 99
    # 99 is the newer benchmark version: OpenML-CC18 Curated Classification benchmark
    for fetched_dataset, openml_dataset, combined_df in get_datasets_by_study(start_index, study_id=99):
        linear_ensemble(fetched_dataset, openml_dataset, combined_df, ag_model_is_plus, groups=["clustering"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='autogluon model folder name', required=True)
    args = parser.parse_args()
    ag_model_is_plus = args.file

    main(ag_model_is_plus)
