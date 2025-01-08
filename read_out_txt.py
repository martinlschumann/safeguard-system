"""Reads out the output from linear-ensembling runs and converts it into a defined datastructure (OpenMLDatasetResult)."""
import ast
import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path


# from https://stackoverflow.com/a/51286749
class EnhancedJSONEncoder(json.JSONEncoder):
    """Encoded dataclasses as json"""

    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


@dataclass()
class OpenMLDatasetResult:
    """The result of one linear-ensembling run."""
    openml_dataset: str  # name of the dataset
    class_name: str  # name of the column of the labels to be inferred
    # contains the accuracy and other metrics of predictor 1 (M1, trained only on labeled data)
    # and predictor 2 (M2, trained on labeled and pseudo-labeled data)
    # example: eval_predictor_1={'accuracy': 0.72, 'balanced_accuracy': 0.5261427837699024, 'mcc': 0.491271579492345}
    eval_predictor_1: dict
    eval_predictor_2: dict
    # how many data items were ignored because their predicted probability was too low
    number_ignored: int
    # how many data items will be unlabeled after the initial split
    initial_unlabeled: int
    # the number of labels in the class to be inferred
    number_of_labels: int
    # output of the safeguard system (might be None on error)
    is_plus: bool | None
    # any other lines that could not be characterized
    other_data: list[str]

    @staticmethod
    def parse(to_parse: list[str]) -> "OpenMLDatasetResult":
        openml_dataset = "EMPTY"
        class_name = "EMPTY"
        eval_predictor_1 = None
        eval_predictor_2 = None
        number_ignored = None
        initial_unlabeled = None
        number_of_labels = None
        other_data = []
        is_plus = None
        for line_to_parse in to_parse:
            if openml_dataset_parse := re.match("openml_dataset='(.+)'\\n", line_to_parse):
                openml_dataset = openml_dataset_parse.group(1)
            elif class_name_parse := re.match("class_name='(.+)'\\n", line_to_parse):
                class_name = class_name_parse.group(1)
            elif eval_predictor_1_parse := re.match("eval_predictor_1=(.+)\\n", line_to_parse):
                eval_predictor_1 = ast.literal_eval(eval_predictor_1_parse.group(1))
            elif eval_predictor_2_parse := re.match("eval_predictor_2=(.+)\\n", line_to_parse):
                eval_predictor_2 = ast.literal_eval(eval_predictor_2_parse.group(1))
            elif number_ignored_parse := re.match("(.+) ignored\\n", line_to_parse):
                number_ignored = int(number_ignored_parse.group(1))
            elif initial_unlabeled_parse := re.match("X_train_2.shape=\\((.+), .+\\)\\n", line_to_parse):
                initial_unlabeled = int(initial_unlabeled_parse.group(1))
            # TODO: this will match twice, as but should only match the correct ones
            elif number_of_labels_parse := re.match("\\(\d+, (\d+)\\)\\n", line_to_parse):
                number_of_labels = int(number_of_labels_parse.group(1))
            elif is_plus_parse := re.match(r"is_plus=(True|False)", line_to_parse):
                # can't use bool(is_plus), because it does not work correctly
                # see https://stackoverflow.com/q/715417
                is_plus = ast.literal_eval(is_plus_parse.group(1))
            else:
                other_data.append(line_to_parse)
        return OpenMLDatasetResult(openml_dataset, class_name, eval_predictor_1, eval_predictor_2, number_ignored,
                                   initial_unlabeled, number_of_labels, is_plus, other_data)

    def to_json(self):
        return json.dumps(self, cls=EnhancedJSONEncoder)

    @staticmethod
    def from_json(json_data: str) -> "OpenMLDatasetResult":
        return OpenMLDatasetResult(**json.loads(json_data))

    @staticmethod
    def from_json_list(json_list: str) -> list["OpenMLDatasetResult"]:
        items = json.loads(json_list)
        output_list = []
        for item in items:
            output_list.append(OpenMLDatasetResult(**item))
        return output_list

    @staticmethod
    def read_json_or_txt(file: str) -> list["OpenMLDatasetResult"]:
        """Reads a json or txt formatted OpenMLDatasetResult file

        :param file: the file to read (requires json or txt extension)
        :return: a OpenMLDatasetResult
        """
        file_path = Path(file)
        extension = file_path.suffix
        if extension == ".txt":
            with open(file_path) as f:
                return convert_output_to_dataclasses(str(file_path))
        elif extension == ".json":
            with open(file_path) as f:
                return OpenMLDatasetResult.from_json_list(f.read())
        else:
            raise ValueError("Extension of file must be txt or json!")

    def get_accuracy(self) -> tuple[float, float]:
        """Returns the accuracies of predictor 1 and 2.

        :return: self.eval_predictor_1["accuracy"], self.eval_predictor_2["accuracy"]
        """
        try:
            return self.eval_predictor_1["accuracy"], self.eval_predictor_2["accuracy"]
        except (AttributeError, TypeError) as e:
            print(f"missing for {self.openml_dataset}")
            raise e


def split_list_by_consecutive_newlines(input_list):
    """Splits list into sub-lists everytime 3 consecutive newlines are encountered,
    which means a new linear-ensembling run.

    Generated with ChatGPT.
    """
    result = []
    current_sublist = []

    for item in input_list:
        if item == "\n" and current_sublist.count("\n") == 2:
            # Three consecutive newlines encountered, start a new sublist
            result.append(current_sublist[:-2])  # Exclude the last two newlines
            current_sublist = []
        else:
            current_sublist.append(item)

    # Add the last sublist if it is not empty
    if current_sublist:
        result.append(current_sublist)

    return result


def convert_output_to_dataclasses(filename="slurm_output/ag.3966274.i23r05c05s04.out.txt") -> list[OpenMLDatasetResult]:
    with open(filename, "r") as f:
        lines: list[str] = f.readlines()
        return convert_lines_to_dataclasses(lines)


def convert_lines_to_dataclasses(lines: list[str]):
    output_start = 0
    for number, line in enumerate(lines):
        if "------------OUTPUT_START------------" in line:
            # as there is one newline after output start, and we want the line after that
            output_start = number + 2
            break
    lines_output = lines[output_start:]
    lines_split = split_list_by_consecutive_newlines(lines_output)
    to_return = []
    for line in lines_split:
        to_return.append(OpenMLDatasetResult.parse(line))
    return to_return


if __name__ == "__main__":
    output = convert_output_to_dataclasses(filename="slurm_output/ag.3971066.i23r05c02s06.out.txt")
    largest_diff = 0
    largest_diff_object = None
    for item in output:
        before, after = item.get_accuracy()
        diff = after - before
        if diff > largest_diff:
            largest_diff = diff
            largest_diff_object = item

        print(
            f"name = {item.openml_dataset} before, after = {item.get_accuracy()}, {item.number_ignored} of {item.initial_unlabeled} ignored ({item.number_ignored / item.initial_unlabeled * 100} %)")

    output_sorted = sorted(output, key=lambda i: i.get_accuracy()[1] - i.get_accuracy()[0])
    print("output sorted\n\n")
    for item in output_sorted:
        print(
            f"name = {item.openml_dataset} before, after = {item.get_accuracy()}, {item.number_ignored} of {item.initial_unlabeled} ignored ({item.number_ignored / item.initial_unlabeled * 100:.2f}%)")
    print(f"{largest_diff=}, name={largest_diff_object.openml_dataset}")
