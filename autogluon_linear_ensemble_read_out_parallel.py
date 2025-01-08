#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It reads data from a slurm output file and only change the is_plus item using the plus_or_minus functions.
Prints the output of this as a json file.
This allows for much quicker full runs, as only the plus_or_minus funcs need to be run.
"""
import argparse
import concurrent.futures
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


# func from chatgpt
def process_dataset(dataset, ag_model_is_plus, groups):
    try:
        fetched_dataset, openml_dataset, combined_df = get_dataset_using_slurm(dataset)
        dataset.is_plus = plus_or_minus_ag(fetched_dataset, combined_df, ag_model_is_plus,
                                           groups=groups)
        return dataset, None
    except Exception as e:
        return None, f"Could not get task with name: {dataset.openml_dataset}, error: {e}"


def main(ag_model_is_plus, slurm_output, output_file, groups):
    datasets: list[OpenMLDatasetResult] = convert_output_to_dataclasses(slurm_output)
    to_output = []
    index_counter = 0
    # from chatgpt
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_dataset, dataset, ag_model_is_plus, groups): dataset for dataset in datasets}
        for future in concurrent.futures.as_completed(futures):
            dataset = futures[future]
            try:
                result, error = future.result()
                if result:
                    to_output.append(result)
                else:
                    print(error)
            except Exception as e:
                print(f"Unexpected error for dataset: {dataset.openml_dataset}, error: {e}")
            index_counter += 1
            print(f"{index_counter} of {len(datasets)}")
    # end chatgpt
    with open(output_file, "w") as output_json:
        output_json.write(json.dumps(to_output, cls=EnhancedJSONEncoder))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='autogluon model folder name', required=True)
    parser.add_argument('-s', '--slurm_output', type=str, help='slurm_output file name', required=True)
    parser.add_argument('-o', '--output', type=str, help='output json file', required=True)
    parser.add_argument('-g', '--groups', type=str, help="pymfe group(s)", nargs='+', default=["model-based"])

    args = parser.parse_args()
    ag_model_is_plus = args.file
    slurm_output = args.slurm_output
    output_file = args.output
    groups = args.groups

    main(ag_model_is_plus, slurm_output, output_file, groups)
