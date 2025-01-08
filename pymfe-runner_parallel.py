#!/usr/bin/env python3.10
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas
from pymfe.mfe import MFE
from sklearn.datasets import fetch_openml
from tqdm import tqdm

from linear_ensemble_basis import time_limit
from read_out_txt import OpenMLDatasetResult


def fetch_dataset(openml_dataset):
    fetched_dataset = fetch_openml(name=openml_dataset, as_frame=True, parser='auto')
    combined_df = fetched_dataset.frame
    return fetched_dataset, openml_dataset, combined_df


def run(dataset, groups):
    try:
        fetched_dataset, openml_dataset, combined_df = fetch_dataset(dataset.openml_dataset)

        print(f"{openml_dataset=}")

        # get accuracy diff and if >=0 save as plus, else minus
        pos_or_neg = "minus"

        if dataset.get_accuracy()[1] - dataset.get_accuracy()[0] >= 0:
            pos_or_neg = "plus"

        class_name = fetched_dataset.target_names[0]

        # mfe requires list or numpy array
        X = combined_df.drop(columns=[class_name]).to_numpy()
        y = combined_df[class_name].to_numpy()

        mfe = MFE(groups=groups)
        # sometimes fit takes too long to run
        with time_limit(10 * 60):  # 10 minutes, same as Autogluon run
            mfe.fit(X, y)
            ft = mfe.extract(out_type=pandas.DataFrame)

        ft["class"] = pos_or_neg
        return ft
    # IndexError e.g. for sylva_agnostic
    except (ValueError, RecursionError, IndexError, TimeoutError) as e:
        print(f"ignoring {dataset.openml_dataset}, reason: {e}")
        return pandas.DataFrame()


def process_datasets(output, groups):
    # created using chatgpt
    results = []
    # Use ProcessPoolExecutor for parallel execution of CPU-bound tasks
    # might error out with memory problems, therefore max workers is set to 10
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(run, dataset, groups): dataset for dataset in output}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing datasets"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Exception encountered: {e}")
    return pandas.concat(results, ignore_index=True)
    # end created by chatgpt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='slurm_output file name',
                        default="../auto-gluon/slurm_output/ag8tpymf.4022621.i23r06c04s12.out.txt")
    parser.add_argument("-j", '--json', type=str, help='json input file name', default=None)
    parser.add_argument('-o', '--output', type=str, help='csv output file name', required=True)
    args = parser.parse_args()

    groups = ["clustering", "model-based"]
    print(f"\n\n{groups=}\n\n")

    if args.json is None:
        file_name = args.input
    else:
        file_name = args.json

    output: list[OpenMLDatasetResult] = OpenMLDatasetResult.read_json_or_txt(file_name)

    total_todo = len(output)
    print(f"Total todo: {total_todo}")

    ft = process_datasets(output, groups)

    # from https://saturncloud.io/blog/how-to-remove-index-column-while-saving-csv-in-pandas/
    ft_csv = ft.to_csv(index=False)
    file_name_csv = args.output
    with open(file_name_csv, "w") as f:
        f.write(ft_csv)
        print(f"wrote to {file_name_csv}")
