#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It reads data from a slurm output file and only change the is_plus item using the plus_or_minus functions.
Prints the output of this as a json file.
This allows for much quicker full runs, as only the plus_or_minus funcs need to be run.
"""
import argparse
import json

from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch

from linear_ensemble_basis import plus_or_minus_ag
from read_out_txt import OpenMLDatasetResult, convert_output_to_dataclasses, EnhancedJSONEncoder


def get_dataset_using_slurm(dataset: OpenMLDatasetResult) -> tuple[Bunch, str, DataFrame]:
    dataset: OpenMLDatasetResult
    name = dataset.openml_dataset
    fetched_dataset = fetch_openml(name=name, as_frame=True, parser='liac-arff')
    combined_df = fetched_dataset.frame.sample(frac=1, random_state=42)
    return fetched_dataset, name, combined_df


def main(ag_model_is_plus, slurm_output, output_file):
    datasets: list[OpenMLDatasetResult] = convert_output_to_dataclasses(slurm_output)
    to_output = []
    index_counter = 0
    for dataset in datasets:
        try:
            fetched_dataset, openml_dataset, combined_df = get_dataset_using_slurm(dataset)
            dataset.is_plus = plus_or_minus_ag(fetched_dataset, combined_df, ag_model_is_plus)
            to_output.append(dataset)
        # NOTE: catch exceptions more individually?
        except Exception as e:
            print(f"Could not get task with name: {dataset.openml_dataset}, error: {e}")

        index_counter += 1
        print(f"{index_counter} of {len(datasets)}")
    with open(output_file, "w") as output_json:
        output_json.write(json.dumps(to_output, cls=EnhancedJSONEncoder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='autogluon model folder name', required=True)
    parser.add_argument('-s', '--slurm_output', type=str, help='slurm_output file name', required=True)
    parser.add_argument('-o', '--output', type=str, help='output json file', required=True)
    args = parser.parse_args()
    ag_model_is_plus = args.file
    slurm_output = args.slurm_output
    output_file = args.output

    main(ag_model_is_plus, slurm_output, output_file)
