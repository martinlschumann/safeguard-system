#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-
"""Calculate the mean accuracy of runs with and without the safeguard system. Also scatter plot the runs with and without the safeguard system."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from adjustText import adjust_text
from pandas import DataFrame

from read_out_txt import OpenMLDatasetResult


def calc_accuracy(output: list[OpenMLDatasetResult]):
    accuracies_with_plus: list[float] = []
    accuracies: list[float] = []
    for result in output:
        if result.is_plus is True:
            accuracies_with_plus.append(result.get_accuracy()[1])
            accuracies.append(result.get_accuracy()[1])
        elif result.is_plus is False:
            accuracies_with_plus.append(result.get_accuracy()[0])
            accuracies.append(result.get_accuracy()[1])
        else:
            # ignore if result.is_plus is None
            pass

    return accuracies_with_plus, accuracies

# from chatgpt
def scatter_plot(output: list[OpenMLDatasetResult]):
    accuracies_with_plus, accuracies = calc_accuracy(output)

    fontsize = 18
    with_title = False

    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(accuracies_with_plus, accuracies, color='blue', label='Dataset')

    # Add diagonal line
    plt.plot([0, 1], [0, 1], color='red', linestyle='-', label="Equal Performance")

    # Add labels for points where accuracies are not equal
    texts = []
    for i, (acc_with_plus, acc) in enumerate(zip(accuracies_with_plus, accuracies)):
        if abs(acc_with_plus - acc) > 0.08:
            texts.append(plt.text(acc_with_plus, acc, output[i].openml_dataset, fontsize=9, ha='right'))

    # Adjust text to avoid overlap
    adjust_text(texts, expand=(1.2, 2), arrowprops=dict(arrowstyle='->', color='gray'))

    # Adding labels and title
    plt.xlabel('Accuracy with Safeguard System', fontsize=fontsize)
    plt.ylabel('Accuracy without Safeguard System', fontsize=fontsize)
    if with_title:
        plt.title('Performance Comparison of Safeguard vs Non-Safeguard Models')

    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)

    # Show plot
    plt.show()


def calculate_mean_accuracy(output: list[OpenMLDatasetResult]):
    accuracies_with_plus, accuracies = calc_accuracy(output)
    if accuracies_with_plus != [] or accuracies != []:
        mean_with_plus = np.mean(accuracies_with_plus)
        mean = np.mean(accuracies)
        return mean_with_plus, mean
    return np.nan, np.nan


def main(file_name, output: list[OpenMLDatasetResult], plot: bool, verbose: bool):
    try:
        mean_with_plus, mean = calculate_mean_accuracy(output)
        if np.isnan(mean_with_plus) or np.isnan(mean):
            if verbose:
                print(f"mean accuracys could not be calculated for {file_name}")
            # should not plot erroneous ones
            return
        if mean_with_plus == mean:
            print(f"mean accuracys for {file_name}: {mean_with_plus=} {mean=}, is same? {mean_with_plus == mean}")
        else:
            print(f"mean accuracys for {file_name}: {mean_with_plus=} {mean=}, is better? {mean_with_plus > mean}")
        if plot:
            scatter_plot(output)
    except Exception as e:
        if verbose:
            print(f"Error {e} with {file_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", "-f", nargs='+', default=[],
                        help="Reads <filenames>.")
    parser.add_argument("--json", "-j", default=[], nargs='+',
                        help="Reads json <filenames>.")
    parser.add_argument("--plot", "-p", default=False, action="store_true")
    parser.add_argument('--verbose', '-v', default=False, action="store_true")
    args = parser.parse_args()

    output: list[OpenMLDatasetResult]
    path: str
    for path in args.filenames:
        output = OpenMLDatasetResult.read_json_or_txt(path)
        file_name = Path(path).name
        main(file_name, output, args.plot, args.verbose)
    for path in args.json:
        output = OpenMLDatasetResult.read_json_or_txt(path)
        file_name = Path(path).name
        main(file_name, output, args.plot, args.verbose)
