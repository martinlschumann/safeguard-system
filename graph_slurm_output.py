#!/usr/bin/env python3.10
import argparse

import matplotlib.pyplot as plt

from read_out_txt import convert_output_to_dataclasses, OpenMLDatasetResult


def get_output_sample(output_sorted, num_samples):
    output_top = output_sorted[-num_samples:]
    output_bottom = output_sorted[:num_samples]
    # division must be integer
    center = len(output_sorted) // 2
    output_center = output_sorted[center:center + num_samples]
    # NOTE: better way to slice this?
    output_sample = []
    output_sample.extend(output_top)
    output_sample.extend(output_bottom)
    output_sample.extend(output_center)
    return output_sample


def color_is_plus(item):
    if item.is_plus is None:
        return "red"
    elif item.is_plus:
        return "orange"
    else:
        return "blue"


def graph_is_plus_diff_accuracy(output_sample, shownames=False):
    x = [i.get_accuracy()[1] - i.get_accuracy()[0] for i in output_sample]
    # y as ratio between total that could be ignored and amount actually ignored.
    y = [i.number_ignored / i.initial_unlabeled * 100 for i in output_sample]
    # labels
    n = [i.openml_dataset for i in output_sample]

    colors = [color_is_plus(i) for i in output_sample]

    # from https://stackoverflow.com/a/14434334
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    plt.axvline(0, color='black')

    # set limits for the graph for better comparison
    # NOTE: big enough? see https://stackoverflow.com/q/11459672 for using x_bound instead which auto scales
    ax.set_xbound(-0.15, 0.1)

    if shownames:
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
    plt.ylabel("Percentage of data ignored")
    plt.xlabel("Change in accuracy (after-before)")
    plt.title("Percentage of data ignored vs change in accuracy (meta feature version)")
    plt.show()


def graph_ignored_vs_accuracy(output_sample, shownames=False):
    x = [i.get_accuracy()[1] for i in output_sample]
    # y as ratio between total that could be ignored and amount actually ignored.
    y = [i.number_ignored / i.initial_unlabeled * 100 for i in output_sample]
    # labels
    n = [i.openml_dataset for i in output_sample]
    colors = ["orange" if i.number_of_labels > 2 else "blue" for i in output_sample]

    # from https://stackoverflow.com/a/14434334
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)

    if shownames:
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
    plt.ylabel("Percentage of data ignored")
    plt.xlabel("Accuracy after pseudo-labeling")
    plt.title("Percentage of data ignored vs accuracy")
    plt.show()


def graph_ignored_vs_diff_accuracy(output_sample, shownames=False):
    x = [i.get_accuracy()[1] - i.get_accuracy()[0] for i in output_sample]
    # y as ratio between total that could be ignored and amount actually ignored.
    y = [i.number_ignored / i.initial_unlabeled * 100 for i in output_sample]
    # labels
    n = [i.openml_dataset for i in output_sample]
    colors = ["orange" if i.number_of_labels > 2 else "blue" for i in output_sample]

    # from https://stackoverflow.com/a/14434334
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    plt.axvline(0, color='black')

    # set limits for the graph for better comparison
    # NOTE: big enough? see https://stackoverflow.com/q/11459672 for using xbound instead which auto scales
    ax.set_xbound(-0.15, 0.1)
    ax.set_ybound(-1, 105)

    if shownames:
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
    plt.ylabel("Percentage of data ignored")
    plt.xlabel("Change in accuracy (after-before)")
    plt.title("Percentage of data ignored vs change in accuracy")
    plt.show()


def graph_ignored_vs_diff_accuracy_is_minus(output_sample, shownames=False):
    x = [acc if (acc := i.get_accuracy()[1] - i.get_accuracy()[0]) > 0 or color_is_plus(i) != "blue" else 0 for i in output_sample]
    # y as ratio between total that could be ignored and amount actually ignored.
    y = [i.number_ignored / i.initial_unlabeled * 100 for i in output_sample]
    # labels
    n = [i.openml_dataset for i in output_sample]
    colors = [color_is_plus(i) for i in output_sample]

    # from https://stackoverflow.com/a/14434334
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=colors)
    plt.axvline(0, color='black')

    # set limits for the graph for better comparison
    # NOTE: big enough? see https://stackoverflow.com/q/11459672 for using xbound instead which auto scales
    ax.set_xbound(-0.15, 0.1)
    ax.set_ybound(0, 100)

    if shownames:
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
    plt.ylabel("Percentage of data ignored")
    plt.xlabel("Change in accuracy (after-before)")
    plt.title("Percentage of data ignored vs change in accuracy (is-minus version)")
    plt.show()


def graph_ignored_vs_diff_accuracy_hist(output_sample, shownames=False):
    x = [i.get_accuracy()[1] - i.get_accuracy()[0] for i in output_sample]

    print(f"{len(x)=}")
    # y as ratio between total that could be ignored and amount actually ignored.
    y = [i.number_ignored / i.initial_unlabeled * 100 for i in output_sample]
    # labels
    n = [i.openml_dataset for i in output_sample]
    colors = ["orange" if i.number_of_labels > 2 else "blue" for i in output_sample]

    # from https://stackoverflow.com/a/14434334
    fig, ax = plt.subplots()
    ax.hist(x)
    plt.axvline(0, color='black')

    # set limits for the graph for better comparison
    # NOTE: big enough? see https://stackoverflow.com/q/11459672 for using xbound instead which auto scales
    ax.set_xbound(-0.15, 0.1)
    ax.set_ybound(0, 120)

    if shownames:
        for i, txt in enumerate(n):
            ax.annotate(txt, (x[i], y[i]))
    plt.ylabel("Amount")
    plt.xlabel("Change in accuracy (after-before)")
    plt.title("Change in accuracy (histogram)")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--amount", "-a", type=int, default=500)
    parser.add_argument("--filename", "-f", type=str, default="slurm_output/ag.3966579+ 3967082.i23r06c01s04.out.txt",
                        help="Reads <filename>.")
    parser.add_argument("--json", "-j", type=str, default=None,
                        help="Reads <filename>.")
    parser.add_argument("--shownames", "-s", default=False, action="store_true")
    parser.add_argument("--verbose", "-v", default=False, action="store_true")
    args = parser.parse_args()

    # threshold of 1/labels => almost never any removed
    # output: list[OpenMLDatasetResult] = convert_output_to_dataclasses(filename="slurm_output/ag.3977947 + 3977947.i23r06c03s04.out.txt")
    # threshold of 0.5
    output: list[OpenMLDatasetResult]
    if args.json is None:
        output = convert_output_to_dataclasses(filename=args.filename)
    else:
        with open(args.json) as json_list:
            output = OpenMLDatasetResult.from_json_list(json_list.read())

    output_sorted = sorted(output, key=lambda i: i.get_accuracy()[1] - i.get_accuracy()[0])
    output_sample = get_output_sample(output_sorted, args.amount)

    if args.verbose:
        print([i.openml_dataset for i in output_sample])

    # graph_ignored_vs_accuracy(output_sample, args.shownames)
    # graph_ignored_vs_diff_accuracy(output_sample, args.shownames)
    # graph_ignored_vs_diff_accuracy_hist(output_sample, args.shownames)
    graph_is_plus_diff_accuracy(output_sample, args.shownames)
    # graph_ignored_vs_diff_accuracy_is_minus(output_sample, args.shownames)
