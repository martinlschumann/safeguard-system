#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Counts which run performed better given 2 slurm_output runs by counting which run won for each dataset."""

import argparse

from read_out_txt import OpenMLDatasetResult


def create_pairs(result_a: list[OpenMLDatasetResult], result_b: list[OpenMLDatasetResult]) -> list[
    tuple[OpenMLDatasetResult, OpenMLDatasetResult]]:
    """Given 2 lists, creates pairs where the name of the datasets is the same.

    :param result_a: the results of run a
    :param result_b: the results of run b
    :return: a tuple of results from run a and b for the same dataset.
    """
    # inspired from chatgpt code
    dict_a: dict[str, OpenMLDatasetResult] = {item.openml_dataset: item for item in result_a}
    dict_b: dict[str, OpenMLDatasetResult] = {item.openml_dataset: item for item in result_b}
    pairs: list[tuple[OpenMLDatasetResult, OpenMLDatasetResult]] = [(dict_a[name], dict_b[name]) for name in dict_a if
                                                                    name in dict_b]
    return pairs
    # end chatgpt code


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Compare 2 output files")
    parser.add_argument("-a", "--first", help="First file to compare, must be json or txt.", required=True, type=str)
    parser.add_argument("-b", "--second", help="Second file to compare, must be json or txt", required=True, type=str)
    parser.add_argument("--before", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("--bva", help="Before from a vs after from b", action="store_true", default=False)
    args = parser.parse_args()

    result_a: list[OpenMLDatasetResult] = OpenMLDatasetResult.read_json_or_txt(args.first)
    result_b: list[OpenMLDatasetResult] = OpenMLDatasetResult.read_json_or_txt(args.second)

    pairs: list[(OpenMLDatasetResult, OpenMLDatasetResult)] = create_pairs(result_a, result_b)
    better_results = []
    for pair in pairs:
        pair: (OpenMLDatasetResult, OpenMLDatasetResult)
        if args.before:
            accuracy_ = pair[0].get_accuracy()[0] < pair[1].get_accuracy()[0]
            equal = pair[0].get_accuracy()[0] == pair[1].get_accuracy()[0]
            if args.verbose:
                print(f"{pair[0].get_accuracy()[0]} {pair[1].get_accuracy()[0]}")
        elif args.bva:
            accuracy_ = pair[0].get_accuracy()[0] < pair[1].get_accuracy()[1]
            equal = pair[0].get_accuracy()[0] == pair[1].get_accuracy()[1]
            if args.verbose:
                print(f"{pair[0].get_accuracy()[0]} {pair[1].get_accuracy()[1]}")
        else:
            accuracy_ = pair[0].get_accuracy()[1] < pair[1].get_accuracy()[1]
            equal = pair[0].get_accuracy()[1] == pair[1].get_accuracy()[1]
            if args.verbose:
                print(f"{pair[0].get_accuracy()[1]} {pair[1].get_accuracy()[1]}")
        if accuracy_:
            better_result = "b"
            better_results.append(better_result)
        elif equal:
            better_result = "ab"
            better_results.append(better_result)
        else:
            better_result = "a"
            better_results.append(better_result)
        if args.verbose:
            print(f"{pair[0].openml_dataset}: better result: {better_result}")
    count_a = better_results.count('a')
    count_b = better_results.count('b')
    count_same = better_results.count('ab')
    print(f"{args.before=}, {args.bva=}")
    print(
        f"\ntotal better: {args.first if count_a >= count_b else args.second}: {count_a} (a) vs {count_b} (b) vs {count_same} (equal)")
